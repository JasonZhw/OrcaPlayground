"""固定基座 G1 的右臂位置控制、夹爪联动及本地运动学求解。"""

import asyncio

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PREFIX = 'unitree_g1_1_'
RIGHT_JOINTS = [PREFIX+'right_'+n+'_joint' for n in (
    'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
    'wrist_roll', 'wrist_pitch', 'wrist_yaw',
)]
GRIPPER_JOINTS = [PREFIX+n for n in (
    'idx71_gripper_r_inner_joint1', 'idx72_gripper_r_inner_joint3',
    'idx73_gripper_r_inner_joint4', 'idx79_gripper_r_inner_joint2',
    'idx81_gripper_r_outer_joint1', 'idx82_gripper_r_outer_joint3',
    'idx83_gripper_r_outer_joint4', 'idx89_gripper_r_outer_joint2',
)]
SITE = PREFIX+'ee_center_site_r'
TOP_DOWN_ROTATION = np.column_stack(([0,0,-1], [0,1,0], [1,0,0]))
HOME_POSITION = np.array([.57, -3.56, 5.08])


def gripper_angle(closure):
    if not np.isfinite(closure) or not 0 <= closure <= 1:
        raise ValueError('夹爪闭合比例必须在 0–1')
    return .65 - float(closure)


class GripperCalibration:
    """按照模型的 connect 约束求四连杆位置，不能假定从动关节同角度。"""

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.indices = np.array([model.joint(n).qposadr[0] for n in GRIPPER_JOINTS])
        self.limits = np.array([model.joint(n).range for n in GRIPPER_JOINTS])
        self.tips = [model.body(PREFIX+'gripper_r_'+b+'_link4').id for b in ('inner','outer')]
        self.equalities = [i for i in range(model.neq)
                           if model.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT
                           and model.eq_obj1id[i] in self.tips]
        if len(self.equalities) != 2:
            raise RuntimeError('右夹爪必须有两组真实四连杆 connect 约束')
        mujoco.mj_kinematics(model, self.data)
        self.rotations = [self.data.xmat[b].reshape(3,3).copy() for b in self.tips]
        self.cache = {}

    def targets(self, closure):
        u = gripper_angle(closure)
        key = round(float(closure), 6)
        if key in self.cache:
            return self.cache[key].copy()
        m, d = self.model, self.data

        def residual(q):
            d.qpos[self.indices] = q
            mujoco.mj_kinematics(m,d)
            errors = [q[0]-u, q[4]-u]
            for i in self.equalities:
                b1,b2 = m.eq_obj1id[i],m.eq_obj2id[i]
                p1 = d.xpos[b1]+d.xmat[b1].reshape(3,3)@m.eq_data[i,:3]
                p2 = d.xpos[b2]+d.xmat[b2].reshape(3,3)@m.eq_data[i,3:6]
                errors.extend(20*(p1-p2))
            for b,rotation in zip(self.tips,self.rotations):
                errors.extend(Rotation.from_matrix(d.xmat[b].reshape(3,3)@rotation.T).as_rotvec())
            return errors

        seed = self.cache[min(self.cache,key=lambda k:abs(k-key))] if self.cache else np.zeros(8)
        solution = least_squares(residual, seed, bounds=(self.limits[:,0],self.limits[:,1]),
                                 max_nfev=120, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if np.linalg.norm(solution.fun) > 1e-4:
            raise RuntimeError('夹爪四连杆目标违反机械约束')
        self.cache[key] = solution.x.copy()
        return solution.x.copy()


def validate_action(action, limits):
    value = np.asarray(action, dtype=float)
    if value.shape != (8,) or not np.isfinite(value).all():
        raise ValueError('动作必须为有限 8 维位置目标')
    limits = np.asarray(limits)
    if np.any(value[:7] < limits[:,0]-1e-6) or np.any(value[:7] > limits[:,1]+1e-6):
        raise ValueError('关节位置目标超过限位')
    if not 0 <= value[7] <= 1:
        raise ValueError('夹爪闭合比例必须在 0–1')
    return value


def interpolate_actions(previous,target,limits,period,*,joint_rate=1.2,gripper_rate=1.6):
    previous=validate_action(previous,limits)
    target=validate_action(target,limits)
    if period<=0 or not np.isfinite(period):
        raise ValueError('控制周期必须有限且大于零')
    rates=np.r_[np.full(7,joint_rate),gripper_rate]
    if not np.isfinite(rates).all() or np.any(rates<=0):
        raise ValueError('动作速率必须有限且大于零')
    count=max(1,int(np.ceil(np.max(np.abs(target-previous)/(period*rates)))))
    for alpha in np.arange(1,count+1)/count:
        yield previous+(target-previous)*alpha


def solve_site_ik(model, qpos, joint_names, site_name, position, rotation, *,
                  position_tolerance=.005, rotation_tolerance=.04, attempts=18,
                  joint_regularization=0.):
    """只修改独立本地副本求解，不在远程场景中瞬移机器人或药盒。"""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    indices = np.array([model.joint(n).qposadr[0] for n in joint_names])
    limits = np.array([model.joint(n).range for n in joint_names])
    site = model.site(site_name).id
    position, rotation = np.asarray(position), np.asarray(rotation)
    if position.shape != (3,) or rotation.shape != (3,3) or not np.isfinite(position).all():
        raise ValueError('目标末端位姿无效')
    rng = np.random.default_rng(0)
    seeds = [data.qpos[indices].copy(), *rng.uniform(limits[:,0], limits[:,1], (attempts-1,len(indices)))]
    reference = data.qpos[indices].copy()

    def residual(q):
        data.qpos[indices] = q
        mujoco.mj_kinematics(model, data)
        angle = Rotation.from_matrix(rotation @ data.site_xmat[site].reshape(3,3).T).as_rotvec()
        geometry = np.r_[data.site_xpos[site]-position, .15*angle]
        return np.r_[geometry, joint_regularization*(q-reference)] if joint_regularization else geometry

    best = None
    for seed in seeds:
        result = least_squares(residual, np.clip(seed, limits[:,0]+1e-6, limits[:,1]-1e-6),
                               bounds=(limits[:,0], limits[:,1]), max_nfev=180,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
        if np.linalg.norm(result.fun[:3]) <= position_tolerance and np.linalg.norm(result.fun[3:6])/.15 <= rotation_tolerance:
            # 位姿容差优先于正则化代价，不能丢弃已找到的可行候选。
            best = result
            break
    pos_error = float(np.linalg.norm(best.fun[:3]))
    angle_error = float(np.linalg.norm(best.fun[3:6])/.15)
    if pos_error > position_tolerance or angle_error > rotation_tolerance:
        raise RuntimeError(f'目标不可达：位置误差 {pos_error:.4f}m，姿态误差 {angle_error:.4f}rad')
    residual(best.x)
    return {'joints': best.x.copy(), 'qpos': data.qpos.copy(),
            'position_error': pos_error, 'rotation_error': angle_error}


def home_joints(robot):
    """固定标定支路，初态不依赖物体真值或随机 IK 分支。"""
    qpos = robot.initial_qpos.copy()
    return solve_site_ik(robot.model, qpos, RIGHT_JOINTS, SITE,
                         HOME_POSITION, TOP_DOWN_ROTATION)['joints']


class FixedBaseRobot:
    """官方 OrcaGymLocal 物理步进，同步实际状态到 OrcaLab 渲染。"""

    def __init__(self, addr='127.0.0.1:50051', *, dt=.001, frame_skip=50):
        import grpc
        from orca_gym.core.orca_gym_local import OrcaGymLocal
        from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.channel = grpc.aio.insecure_channel(addr, options=[('grpc.max_receive_message_length',1024**3)])
        self.gym = OrcaGymLocal(GrpcServiceStub(self.channel))
        self.loop.run_until_complete(self.gym.pause_simulation())
        self.xml_path = self.loop.run_until_complete(self.gym.load_model_xml())
        self.loop.run_until_complete(self.gym.init_simulation(self.xml_path))
        self.model, self.data = self.gym._mjModel, self.gym._mjData
        self.model.opt.timestep = dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # 同一个四连杆内的网格在转轴处相交；豁免机械相连零件的自碰撞。
        # 内外两指之间、手指与药盒/桌面之间的接触全部保留。
        self.collision_exemptions = []
        geom_chain = np.zeros(self.model.ngeom,dtype=np.int8)
        for side in ('l','r'):
            for branch in ('inner','outer'):
                chain = {self.model.body(PREFIX+f'gripper_{side}_{branch}_link{i}').id
                         for i in range(1,5)}
                self.collision_exemptions.append(chain)
                for g in range(self.model.ngeom):
                    if self.model.geom_bodyid[g] in chain:
                        geom_chain[g]=len(self.collision_exemptions)
        def contact_filter(model,data,g1,g2):
            return int(geom_chain[g1]>0 and geom_chain[g1]==geom_chain[g2])
        self._contact_filter = contact_filter
        mujoco.set_mjcb_contactfilter(contact_filter)
        self.dt, self.frame_skip, self.sim_time = dt, frame_skip, 0.
        self.joints = self.gym.model.get_joint_dict()
        self.actuators = self.gym.model.get_actuator_dict()
        if any(n.startswith(PREFIX) and j['Type'] == 0 for n,j in self.joints.items()):
            self.disconnect()
            raise RuntimeError('G1 基座尚未固定；请加载 fixed_base 场景')
        self.arm_qpos = np.array([self.joints[n]['QposIdxStart'] for n in RIGHT_JOINTS])
        self.arm_act = np.array([self.model.actuator(n).id for n in RIGHT_JOINTS])
        self.gripper_act = np.array([self.model.actuator(n+'_pctrl').id for n in GRIPPER_JOINTS])
        self.limits = np.array([self.joints[n]['Range'] for n in RIGHT_JOINTS])
        self.initial_qpos = self.data.qpos.copy()
        self.ctrl = np.array([self.initial_qpos[self.joints[a['JointName']]['QposIdxStart']]
                              for a in self.actuators.values()])
        self.gripper = GripperCalibration(self.model)
        # 隐式积分处理关节阻尼；夹爪保留力限位，实际夹持仍由接触完成。
        names = RIGHT_JOINTS + [n+'_pctrl' for n in GRIPPER_JOINTS]
        for name in names:
            arm = name in RIGHT_JOINTS
            kp, kv = (120., 8.) if arm else (20., .5)
            a = self.model.actuator(name).id
            self.model.actuator_gainprm[a,0] = kp
            self.model.actuator_biasprm[a,:3] = [0.,-kp,-kv]
        # connect 约束已经联动 joint2/joint3；重复位置伺服会与被药盒挡住的
        # 夹爪互相对抗。仅驱动两侧 joint1，joint4 保持指面平行，其余被动。
        for local_idx in (1,3,5,7):
            a = self.gripper_act[local_idx]
            self.model.actuator_gainprm[a,0] = 0.
            self.model.actuator_biasprm[a,:3] = [0.,0.,-.03]
        for a in range(self.model.nu):
            if 'waist' in self.model.actuator(a).name:
                self.model.actuator_gainprm[a,0] = 400.
                self.model.actuator_biasprm[a,:3] = [0.,-400.,20.*-1]
        self.actuator_dofs = self.model.jnt_dofadr[self.model.actuator_trnid[:,0]]
        self.gym.set_protected_override_ctrl_ids(range(self.model.nu))
        mujoco.mj_forward(self.model,self.data)
        self.base_reference = self.body_pose(PREFIX+'pelvis')['Pos'].copy()

    def refresh(self):
        self.gym.update_data()

    def state(self):
        self.refresh()
        q = self.data.qpos
        grip = float(np.clip(.65-q[self.joints[GRIPPER_JOINTS[0]]['QposIdxStart']], 0, 1))
        return np.r_[q[self.arm_qpos], grip].astype(np.float32)

    def body_pose(self, name):
        poses = self.gym.query_body_xpos_xmat_xquat([name])
        return {k: np.asarray(v) for k,v in poses[name].items()}

    def site_pose(self):
        site = self.model.site(SITE).id
        return {'xpos':self.data.site_xpos[site].copy(),'xmat':self.data.site_xmat[site].reshape(3,3).copy()}

    def step(self, action):
        action = validate_action(action, self.limits)
        self.ctrl[self.arm_act] = action[:7]
        self.ctrl[self.gripper_act] = 0.
        self.ctrl[self.gripper_act[[0,4]]] = gripper_angle(action[7])
        actual_gripper = self.data.qpos[self.gripper.indices]
        self.ctrl[self.gripper_act[2]] = actual_gripper[0]-actual_gripper[1]
        self.ctrl[self.gripper_act[6]] = -actual_gripper[4]-actual_gripper[5]
        before = self.data.time
        # 重力补偿仍通过执行器施力，不改 qpos；训练动作维持为绝对关节目标。
        gains = self.model.actuator_gainprm[:,0]
        gravity = np.divide(self.data.qfrc_bias[self.actuator_dofs],gains,
                            out=np.zeros_like(gains),where=gains>0)
        self.gym.set_ctrl(self.ctrl+gravity)
        self.gym.mj_step(self.frame_skip)
        self.sim_time = self.data.time
        if abs(self.sim_time-before-self.dt*self.frame_skip) > 1e-6 or np.any(self.data.warning.number):
            raise RuntimeError('物理步进出现数值警告或自动重置')
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('物理状态出现非有限值')
        if np.linalg.norm(self.body_pose(PREFIX+'pelvis')['Pos']-self.base_reference) > 1e-5:
            raise RuntimeError('固定基座发生位移')
        return self.state()

    def render(self, index=-1):
        self.refresh()
        self.loop.run_until_complete(self.gym.render(simulate_index=index))

    def reset(self, *, arm_home=None, closure=0):
        mujoco.mj_resetData(self.model,self.data)
        if arm_home is not None:
            self.data.qpos[self.arm_qpos] = arm_home
        self.data.qpos[self.gripper.indices] = self.gripper.targets(closure)
        mujoco.mj_forward(self.model,self.data)
        self.ctrl[:] = np.array([self.data.qpos[self.joints[a['JointName']]['QposIdxStart']]
                                 for a in self.actuators.values()])
        self.sim_time = 0.

    def disconnect(self):
        # 保持被动暂停，避免专家退出后无人控制的仿真继续运行。
        self.loop.run_until_complete(self.channel.close())
        self.loop.close()
        mujoco.set_mjcb_contactfilter(None)
