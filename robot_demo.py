"""Japanese instruction -> pretrained semantic model -> simulated arm.
The object positions are known, not estimated from camera images.
"""
import json
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
COLORS = ['red', 'blue', 'yellow']
DESCRIPTIONS = ['赤い箱を指してください。', '青い箱を指してください。', '黄色い箱を指してください。']
POSITIONS = [[0.48, -0.26, 0.06], [0.63, 0.0, 0.06], [0.48, 0.26, 0.06]]
RGBA = [[0.9, 0.08, 0.08, 1], [0.08, 0.25, 0.95, 1], [0.95, 0.8, 0.05, 1]]
DEFAULT_COMMANDS = ['赤色の箱を指して', '青い箱の上に移動して', '黄色の箱を示して']

class SemanticSelector:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_ID, device='cpu')
        self.candidates = self.model.encode(DESCRIPTIONS, normalize_embeddings=True)

    def select(self, command):
        query = self.model.encode([command], normalize_embeddings=True)[0]
        scores = self.candidates @ query
        index = int(np.argmax(scores))
        return index, {c: float(s) for c, s in zip(COLORS, scores)}

def get_font(size):
    for path in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def run_demo(selector, commands=None, output_dir='pai_output', fps=15, width=640, height=480):
    commands = list(DEFAULT_COMMANDS if commands is None else commands)
    if not commands:
        raise ValueError('指示を1つ以上指定してください。')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    decisions = []
    for command in commands:
        index, scores = selector.select(command)
        decisions.append({'command': command, 'selected': COLORS[index],
                          'index': index, 'similarities': scores})
        print(command, '->', COLORS[index], {k: round(v, 3) for k,v in scores.items()})
    client = p.connect(p.DIRECT)
    writer = None
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        p.loadURDF('plane.urdf')
        robot = p.loadURDF('kuka_iiwa/model.urdf', useFixedBase=True)
        joints = [j for j in range(p.getNumJoints(robot))
                  if p.getJointInfo(robot,j)[2] != p.JOINT_FIXED]
        for pos, rgba in zip(POSITIONS, RGBA):
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[.065]*3, rgbaColor=rgba)
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.065]*3)
            p.createMultiBody(0, collision, visual, basePosition=pos)
        marker = p.createMultiBody(0, -1, p.createVisualShape(
            p.GEOM_SPHERE, radius=.018, rgbaColor=[0,1,0,1]), [0,0,-2])
        home = np.array([.35, 0, .65])
        orientation = p.getQuaternionFromEuler([0, np.pi, 0])
        def ik(pos):
            return p.calculateInverseKinematics(robot, 6, pos, orientation,
                                                maxNumIterations=150, residualThreshold=1e-5)
        for j, angle in zip(joints, ik(home)):
            p.resetJointState(robot,j,angle)
        view = p.computeViewMatrixFromYawPitchRoll([.3,0,.45],2.1,55,-25,0,2)
        projection = p.computeProjectionMatrixFOV(48,width/height,.05,5)
        font, small = get_font(21), get_font(16)
        video = out / 'robot_demo.mp4'
        writer = imageio.get_writer(str(video), fps=fps, codec='libx264',
                                   quality=7, macro_block_size=16)
        count = 0
        def capture(title, subtitle, details=''):
            nonlocal count
            rgb = np.asarray(p.getCameraImage(width,height,view,projection,
                renderer=p.ER_TINY_RENDERER)[2],dtype=np.uint8).reshape(height,width,4)[:,:,:3]
            im = Image.fromarray(rgb)
            draw = ImageDraw.Draw(im)
            draw.rectangle([0,0,width,100], fill=(20,25,35))
            draw.text((12,8),title,font=font,fill='white')
            draw.text((12,40),subtitle,font=small,fill='white')
            draw.text((12,68),details,font=small,fill=(180,220,255))
            draw.rectangle([0,height-34,width,height],fill=(20,25,35))
            draw.text((12,height-28),'AI: multilingual MiniLM | Control: IK | Simulation: PyBullet',
                      font=small,fill='white')
            writer.append_data(np.asarray(im))
            if count == 0:
                im.save(out/'preview.png')
            count += 1
        def move(start, end, seconds, title, subtitle, details=''):
            for frame in range(round(seconds*fps)):
                u=(frame+1)/round(seconds*fps)
                blend=u*u*(3-2*u)
                target=(1-blend)*np.asarray(start)+blend*np.asarray(end)
                p.setJointMotorControlArray(robot,joints,p.POSITION_CONTROL,
                    targetPositions=ik(target), forces=[300]*len(joints))
                for _ in range(round(240/fps)):
                    p.stepSimulation()
                capture(title,subtitle,details)
        move(home,home,5,'自然言語で対象を選ぶロボットアーム',
             '学習済み文章モデルで指示と3つの候補文を比較する',
             '対象位置は既知 / 画像認識・把持・追加学習は行わない')
        for n, result in enumerate(decisions,1):
            target=np.array(POSITIONS[result['index']])+[0,0,.21]
            p.resetBasePositionAndOrientation(marker,target,[0,0,0,1])
            title=f"{n}/{len(decisions)} 指示：{result['command']}"
            subtitle=f"AIの選択：{result['selected']}（緑の点が目標）"
            scores='類似度 '+ ' / '.join(f'{k}: {v:.3f}' for k,v in result['similarities'].items())
            move(home,home,3,title,subtitle,scores)
            move(home,target,5,title,subtitle,scores)
            move(target,target,6,title,subtitle,scores)
            actual=np.array(p.getLinkState(robot,6,computeForwardKinematics=True)[4])
            error=float(np.linalg.norm(actual-target))
            result.update(target_xyz=target.tolist(), actual_xyz=actual.tolist(),
                          position_error_m=error, reached=error<.03)
            print(f"  arm error: {error:.4f} m, reached: {result['reached']}")
            move(target,home,5,title,'待機位置へ戻る',scores)
        passed=sum(r['reached'] for r in decisions)
        move(home,home,8,'結果と役割分担',f'目標位置への到達：{passed}/{len(decisions)} 回（誤差3cm未満）',
             'AI：対象の選択 / 逆運動学と位置制御：アームの移動')
        writer.close()
        writer=None
        (out/'results.json').write_text(json.dumps({'model':MODEL_ID,
            'candidate_descriptions':DESCRIPTIONS,'fps':fps,'frames':count,
            'duration_s':count/fps,'results':decisions},ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'動画: {video}, 長さ: {count/fps:.1f} 秒')
        return str(video), decisions
    finally:
        if writer is not None:
            writer.close()
        p.disconnect(client)

if __name__ == '__main__':
    commands = json.loads(Path('commands.json').read_text(encoding='utf-8')) if Path('commands.json').exists() else None
    run_demo(SemanticSelector(), commands)
