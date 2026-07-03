# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List, Tuple, Any

from curobo.types.math import Pose
from curobo.types.state import JointState as CuJointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from curobo.wrap.reacher.motion_gen import MotionGenStatus
from curobo.wrap.reacher.motion_gen import PoseCostMetric
from isaac_ros_cumotion.cumotion_planner import CumotionActionServer
from isaac_ros_cumotion_interfaces.action import MotionPlan
from moveit_msgs.msg import MoveItErrorCodes
import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor

from curobo.geom.types import Mesh, Obstacle
from curobo.geom.sphere_fit import fit_spheres_to_mesh
import torch
import numpy as np

from geometry_msgs.msg import Point, Quaternion
from scipy.spatial.transform import Rotation as R
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


class CumotionGoalSetPlannerServer(CumotionActionServer):

    def __init__(self):

        super().__init__()
        self.locked_joint_positions = self.robot_config["kinematics"]["lock_joints"]
        
        self._goal_set_planner_server = ActionServer(
            self, MotionPlan, "cumotion/motion_plan", self.motion_plan_execute_callback
        )
        self.scene_hash = "" # usefull for only updating scene when changes happen identified easily with a hash
        self._attached_spheres_tensor = None
        
    def reset_planner(self):
        pass
    #     try:
    #         self.motion_gen.reset()
    #     except Exception as e:
    #         self.get_logger().warn(f"Reset failed: {e}")
            
    def warmup(self):
        self.get_logger().info("warming up cuMotion, wait until ready")
        # self.motion_gen.warmup(enable_graph=True, n_goalset=100, warmup_js_trajopt=True)
        self.get_logger().info("cuMotion is ready for planning queries!")

    def toggle_link_collision(self, collision_link_names: List[str], enable_flag: bool):
        if len(collision_link_names) > 0:
            if enable_flag:
                for k in collision_link_names:
                    self.motion_gen.kinematics.kinematics_config.enable_link_spheres(k)
            else:
                for k in collision_link_names:
                    self.motion_gen.kinematics.kinematics_config.disable_link_spheres(k)

    def get_cu_pose_from_ros_pose(self, ros_pose):
        cu_pose = Pose.from_list(
            [
                ros_pose.position.x,
                ros_pose.position.y,
                ros_pose.position.z,
                ros_pose.orientation.w,
                ros_pose.orientation.x,
                ros_pose.orientation.y,
                ros_pose.orientation.z,
            ]
        )
        return cu_pose

    def get_goal_poses(self, plan_req: MotionPlan.Goal) -> Pose:
        if plan_req.goal_pose.header.frame_id != self.motion_gen.kinematics.base_link:
            self.get_logger().error(
                "Planning frame: "
                + plan_req.goal_pose.header.frame_id
                + " is not same as motion gen frame: "
                + self.motion_gen.kinematics.base_link
            )
            return False, MoveItErrorCodes.INVALID_LINK_NAME, []
        poses = []
        for k in plan_req.goal_pose.poses:
            poses.append(
                [
                    k.position.x,
                    k.position.y,
                    k.position.z,
                    k.orientation.w,
                    k.orientation.x,
                    k.orientation.y,
                    k.orientation.z,
                ]
            )
        if len(poses) == 0:
            self.get_logger().error("No goal pose found")
            return False, MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS, poses
        goal_pose = Pose.from_batch_list(poses, self.motion_gen.tensor_args)
        goal_pose = Pose(
            position=goal_pose.position.view(1, -1, 3),
            quaternion=goal_pose.quaternion.view(1, -1, 4),
        )
        return True, MoveItErrorCodes.SUCCESS, goal_pose

    def _handle_scene(self,goal_handle: MotionPlan.Goal) -> Tuple[MotionPlan.Result, torch.Tensor, Exception]:
        
        result = MotionPlan.Result()
        result.success = False
        
        total_spheres_tensor = None
        
        if not goal_handle.request.use_planning_scene:
            print("not using planning scene")
            return result, total_spheres_tensor, None

        ### Scene Handling
        new_scene_hash = (
            hash(str(goal_handle.request.robot_state.attached_collision_objects))
            + hash(str(goal_handle.request.use_planning_scene))
            + hash(str(goal_handle.request.world.octomap))
        )
        print("Old scene hash: ",self.scene_hash )
        print("Scene hash:     ",new_scene_hash, "number of attached objects: ", len(goal_handle.request.robot_state.attached_collision_objects) if goal_handle.request.robot_state else "no robot state in request")
        
        if  new_scene_hash != self.scene_hash:
            self.scene_hash = new_scene_hash
            self.get_logger().info("New scene identified updating curobo...")
            
            total_spheres = []
            if goal_handle.request.use_planning_scene:
                self.get_logger().info("Updating planning scene")
                scene = goal_handle.request.world
                world_objects = scene.collision_objects
                world_update_status = self.update_world_objects(world_objects)
                if not world_update_status:
                    result.success = False
                    result.error_code.val = MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE
                    self.get_logger().error("World update failed.")
                    return result,None, Exception("World update failed.")
                
                # 2. Extract and add attached collision objects (tools, grasped items)
                # This is the missing step! The goal message puts them here.
                if (goal_handle.request.robot_state and 
                    goal_handle.request.robot_state.attached_collision_objects):
                    
                    num_attached = len(goal_handle.request.robot_state.attached_collision_objects)
                    self.get_logger().info(f"Found {num_attached} attached object(s) in robot_state.")
                    
                    allocated_curr_spheres = self.motion_gen.robot_cfg.kinematics.kinematics_config.get_link_spheres("gripper")
                    self.get_logger().info(f"Current gripper spheres before adding attached objects: {allocated_curr_spheres.shape}")
                    
                    # handling spheres for attached objects
                    # self.motion_gen.detach_spheres_from_robot(link_name="gripper")
                    
                    attached_meshes_num = 0
                    for attached_obj in goal_handle.request.robot_state.attached_collision_objects:
                        attached_meshes_num += len(attached_obj.object.meshes)
                    self.get_logger().info(f"Total attached meshes to process: {attached_meshes_num}")
                    
                    
                    surface_sphere_radius = 0.01  # 1 cm
                    
                    for attached_obj in goal_handle.request.robot_state.attached_collision_objects:
                        pose = attached_obj.object.pose
                        for i, mesh in enumerate(attached_obj.object.meshes):
                            self.get_logger().info(f"Processing attached object id: {attached_obj.object.id}, mesh index: {i}")
                            mesh_vertices = mesh.vertices
                            mesh_triangles = mesh.triangles
                            faces = []
                            for tri in mesh_triangles:
                                faces.append([tri.vertex_indices[0], tri.vertex_indices[1], tri.vertex_indices[2]])
                            faces = torch.tensor(faces)
                            points = []
                            for v in mesh_vertices:
                                points.append([v.x, v.y, v.z])
                            points = torch.tensor(points, dtype=torch.float32)

                            position: Point = pose.position
                            orientation: Quaternion = pose.orientation
                            self.get_logger().info(f"Mesh pose position: {position}, orientation: {orientation}")
                            # Position
                            t = np.array([position.x, position.y, position.z], dtype=np.float64)
                            # Quaternion geometry_msgs uses: x, y, z, w
                            quat = np.array([orientation.x, orientation.y, orientation.z, orientation.w],dtype=np.float64)
                            # Convert quaternion to rotation matrix
                            rot = R.from_quat(quat).as_matrix()  # shape (3,3)

                            # Build 4x4 homogeneous transform
                            T = np.eye(4, dtype=np.float64)
                            T[:3, :3] = rot
                            T[:3, 3] = t
                            

                            spheres , radius = fit_spheres_to_mesh(Mesh(name=attached_obj.object.id,vertices=points, faces=faces).get_trimesh_mesh().apply_transform(matrix=T), n_spheres=allocated_curr_spheres.shape[0]//attached_meshes_num + 1, surface_sphere_radius=surface_sphere_radius)
                            self.get_logger().info(f"Fitted {len(spheres)} spheres to attached object {attached_obj.object.id} with radius {radius}")
                            # for each of the spheres we need to have x y z r, the spheres returned are x y z only and radius is np array of radiuses
                            # stack them with numpy
                            spheres = np.hstack([spheres, radius[:, np.newaxis]])
                            total_spheres.extend(spheres)
                    # limit to max the number of spheres allocated
                    max_spheres = allocated_curr_spheres.shape[0]
                    if len(total_spheres) > max_spheres:
                        # make sure the shape are same
                        self.get_logger().warn(f"Number of spheres {len(total_spheres)} for attached objects exceeded max allocated {max_spheres}, truncating.")
                        total_spheres = total_spheres[:max_spheres]
                        self.get_logger().info(f"Truncated to {len(total_spheres)} spheres.")
                        # pad with zeros if less
                    elif len(total_spheres) < max_spheres:
                        self.get_logger().info(f"Number of spheres {len(total_spheres)} for attached objects less than max allocated {max_spheres}, padding with zeros.")
                        num_to_pad = max_spheres - len(total_spheres)
                        for _ in range(num_to_pad):
                            total_spheres.append([0.0, 0.0, 0.0, 0.0])  # x,y,z,radius zero padding
                    
                    self.get_logger().info(f"Spheres data:")
                    for s in total_spheres:
                        self.get_logger().info(f"  Sphere center: {s[:3]}, radius: {s[3]}")
                                
                    total_spheres_np = np.array(total_spheres, dtype=np.float32)        
                    total_spheres_tensor = torch.from_numpy(total_spheres_np)
                    self.get_logger().info(f"Attaching tensor of shape {total_spheres_tensor.shape} to gripper link.")
                    # self.toggle_link_collision(plan_req.disable_collision_links, True)
                    self._attached_spheres_tensor = total_spheres_tensor
                    
            # result.success = False
            # return result
        else:
            self.get_logger().info("Scipping world update.")
            total_spheres_tensor = self._attached_spheres_tensor
        
        return result, total_spheres_tensor, None

    def motion_plan_execute_callback(self, goal_handle):
        import time
        start = time.time()
        
        self.get_logger().info("Executing goal...")
        pose_cost_metric = None
        # check moveit scaling factors:
        time_dilation_factor = goal_handle.request.time_dilation_factor
        if time_dilation_factor == 0.0:
            time_dilation_factor = 0.1
            self.get_logger().warn("Cannot set time_dilation_factor = 0.0")
        self.get_logger().info(
            "Planning with time_dilation_factor: " + str(time_dilation_factor)
        )
        
        plan_req = goal_handle.request
        result = MotionPlan.Result()

        goal_handle.succeed()
        # self.motion_gen.reset(reset_seed=False)

        ### Lock Liftkit Joint (must come before scene update and JS capture)
        if len(plan_req.start_state.name) == 0 or plan_req.start_state.name[0] != "liftkit_joint":
            print("the planning request must contain liftkit_joint at [0] position")
            result.error_code.val = MoveItErrorCodes.START_STATE_INVALID
            result.message = "Liftkit joint name is not in the correct position, expected at 0 index."
            self.reset_planner()
            return result

        requested_locked_joint_positions = {}
        print("plan_req.start_state.name", plan_req.start_state.name)
        print("plan_req.start_state.position", plan_req.start_state.position)

        plan_req_joint_name_to_index_map = {name: idx for idx, name in enumerate(plan_req.start_state.name)}
        for name in self.locked_joint_positions.keys():
            idx = plan_req_joint_name_to_index_map.get(name, None)
            if idx is not None:
                requested_locked_joint_positions[name] = float(plan_req.start_state.position[idx])
                print(f"Requested locked joint: {name} with position {requested_locked_joint_positions[name]}")

        if requested_locked_joint_positions:
            print(f"Requested locked joint positions: {requested_locked_joint_positions}")
            diff = sum(
                abs(
                    float(requested_locked_joint_positions[jn]) -
                    float(self.locked_joint_positions.get(jn, requested_locked_joint_positions[jn]))
                )
                for jn in requested_locked_joint_positions
            )
            if diff > 0.001:
                self.locked_joint_positions.update(requested_locked_joint_positions)
                print(f"Updating locked joints with positions: {self.locked_joint_positions}")
                self.motion_gen.update_locked_joints(
                    self.locked_joint_positions,
                    robot_config_dict=self.robot_config
                )

        print("before scene: ", time.time() - start)
        result, total_spheres_tensor, err = self._handle_scene(goal_handle)
        if isinstance(err, Exception):
            self.get_logger().error("Exception on handle scene: ", err)
            result.error_code.val = MoveItErrorCodes.START_STATE_INVALID
            result.message = "Scene is not available."
            self.reset_planner()
            return result
        print("after scene: ", time.time() - start)

        ### Handle Start State (AFTER scene update so joint state is fresh relative to the updated world)
        start_state = None
        if plan_req.use_current_state:
            if self._CumotionActionServer__js_buffer is None:
                self.get_logger().error(
                    "joint_state was not received from "
                    + self._CumotionActionServer__joint_states_topic
                )
                result.error_code.val = MoveItErrorCodes.START_STATE_INVALID
                result.message = "Current joint state is not available."
                self.reset_planner()
                return result
            state = CuJointState.from_position(
                position=self.tensor_args.to_device(
                    self._CumotionActionServer__js_buffer["position"]
                ).unsqueeze(0),
                joint_names=self._CumotionActionServer__js_buffer["joint_names"],
            )
            state.velocity = self.tensor_args.to_device(
                self._CumotionActionServer__js_buffer["velocity"]
            ).unsqueeze(0)
            start_state = self.motion_gen.get_active_js(state)
            # Do NOT clear __js_buffer — js_callback keeps it current continuously
        elif len(plan_req.start_state.position) > 0:
            start_state = self.motion_gen.get_active_js(
                CuJointState.from_position(
                    position=self.tensor_args.to_device(
                        plan_req.start_state.position
                    ).unsqueeze(0),
                    joint_names=plan_req.start_state.name,
                )
            )
        else:
            self.get_logger().error("joint state in start state was empty")
            result.error_code.val = MotionGenStatus.INVALID_START_STATE_UNKNOWN_ISSUE
            result.message = "Start state is not available."
            self.reset_planner()
            return result
        print("start state: ", time.time() - start)

        if plan_req.plan_grasp:
            self.get_logger().info(
                "Planning to Grasp Object with stop at offset distance"
            )
            success, error_code, poses = self.get_goal_poses(plan_req)
            self.get_logger().info(f"Success, Error Code): {success}, {error_code}!")
            if not success:
                result.error_code.val = error_code
                result.message = "Failed to get goal poses."
                self.reset_planner()
                return result
            hold_vec_weight = None
            if len(plan_req.grasp_partial_pose_vec_weight) == 6:
                hold_vec_weight = [
                    plan_req.grasp_partial_pose_vec_weight[i] for i in range(6)
                ]

            retract_vec_weight = None
            if len(plan_req.retract_partial_pose_vec_weight) == 6:
                retract_vec_weight = [
                    plan_req.retract_partial_pose_vec_weight[i] for i in range(6)
                ]

            grasp_constraint_in_goal_frame = (
                plan_req.grasp_approach_constraint_in_goal_frame
            )
            retract_constraint_in_goal_frame = plan_req.retract_constraint_in_goal_frame
            if total_spheres_tensor!= None: self.motion_gen.attach_spheres_to_robot(None, total_spheres_tensor, link_name="gripper")
            grasp_plan_result = self.motion_gen.plan_grasp(
                start_state,
                poses,
                MotionGenPlanConfig(
                    max_attempts=self._CumotionActionServer__max_attempts,
                    enable_graph_attempt=1,
                    time_dilation_factor=time_dilation_factor,
                ),
                grasp_approach_offset=self.get_cu_pose_from_ros_pose(
                    plan_req.grasp_offset_pose
                ),
                grasp_approach_path_constraint=hold_vec_weight,
                retract_offset=self.get_cu_pose_from_ros_pose(
                    plan_req.retract_offset_pose
                ),
                retract_path_constraint=retract_vec_weight,
                disable_collision_links=plan_req.disable_collision_links,
                plan_approach_to_grasp=plan_req.plan_approach_to_grasp,
                plan_grasp_to_retract=plan_req.plan_grasp_to_retract,
                grasp_approach_constraint_in_goal_frame=grasp_constraint_in_goal_frame,
                retract_constraint_in_goal_frame=retract_constraint_in_goal_frame,
            )
            if grasp_plan_result.success.item():
                traj = self.get_joint_trajectory(
                    grasp_plan_result.grasp_trajectory,
                    grasp_plan_result.grasp_trajectory_dt,
                )
                result.planning_time = grasp_plan_result.planning_time
                result.planned_trajectory.append(traj)
                if plan_req.plan_grasp_to_retract:
                    traj = self.get_joint_trajectory(
                        grasp_plan_result.retract_trajectory,
                        grasp_plan_result.retract_trajectory_dt,
                    )
                    result.planned_trajectory.append(traj)
                result.success = True
                result.goal_index = grasp_plan_result.goalset_index.item()
            else:
                result.success = False
                result.message = grasp_plan_result.status
        else:
            if plan_req.plan_cspace:
                self.get_logger().info("Planning CSpace target")
                if len(plan_req.goal_state.position) <= 0:
                    self.get_logger().error("goal state is empty")
                    result.error_code.val = MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED
                    result.message = "Goal state is not available."
                    self.reset_planner()
                    return result
                
                # result.success = False
                # return result
                goal_state = self.motion_gen.get_active_js(
                    CuJointState.from_position(
                        position=self.tensor_args.to_device(
                            plan_req.goal_state.position
                        ).unsqueeze(0),
                        joint_names=plan_req.goal_state.name,
                    )
                )
                print("plan_req.disable_collision_links",plan_req.disable_collision_links)
                self.toggle_link_collision(plan_req.disable_collision_links, False)
                if total_spheres_tensor!= None: self.motion_gen.attach_spheres_to_robot(None, total_spheres_tensor, link_name="gripper")
                
                print("start state pos: ",start_state.position)
                print("goal state pos: ",goal_state.position)
                motion_gen_result = self.motion_gen.plan_single_js(
                    start_state,
                    goal_state,
                    MotionGenPlanConfig(
                        max_attempts=self._CumotionActionServer__max_attempts,
                        enable_graph_attempt=1,
                        time_dilation_factor=time_dilation_factor,
                    ),
                )
                self.toggle_link_collision(plan_req.disable_collision_links, True)

            elif plan_req.plan_pose:
                print("pose start: ",time.time() - start)
                
                self.get_logger().info("Planning Pose target")
                if plan_req.hold_partial_pose:
                    if len(plan_req.hold_partial_pose_vec_weight) < 6:
                        self.get_logger().error(
                            "Partial pose vec weight should be of length 6"
                        )
                        self.reset_planner()
                        return result

                    hold_vec_weight = [
                        plan_req.hold_partial_pose_vec_weight[i] for i in range(6)
                    ]
                    pose_cost_metric = PoseCostMetric(
                        hold_partial_pose=True,
                        hold_vec_weight=self.motion_gen.tensor_args.to_device(
                            hold_vec_weight
                        ),
                    )

                # read goal poses:
                success, error_code, poses = self.get_goal_poses(plan_req)
                if not success:
                    result.error_code.val = error_code
                    result.message = "Failed to get goal poses."
                    self.reset_planner()
                    return result
                print("after get goal posees: ",time.time() - start)
                

                self.toggle_link_collision(plan_req.disable_collision_links, False)
                
                print("after toggle link collision: ",time.time() - start)
                
                if poses.shape[1] == 1:
                    poses.position = poses.position.view(-1, 3)
                    poses.quaternion = poses.quaternion.view(-1, 4)
                    self.get_logger().error(
                        f"Hi max attempts {self._CumotionActionServer__max_attempts}"
                    )
                    self.get_logger().error(f"time dilation {time_dilation_factor}")
                    if total_spheres_tensor!= None: 
                        print("spheres start: ",time.time() - start)
                        self.motion_gen.attach_spheres_to_robot(None, total_spheres_tensor, link_name="gripper")
                        print("spheres end: ",time.time() - start)
                    print("start motion gen: ",time.time() - start)
                    motion_gen_result = self.motion_gen.plan_single(
                        start_state,
                        poses,
                        MotionGenPlanConfig(
                            # max_attempts=1,
                            # enable_graph_attempt=0,
                            # max_attempts=self._CumotionActionServer__max_attempts,
                            # enable_graph_attempt=1,
                            time_dilation_factor=time_dilation_factor,
                            pose_cost_metric=pose_cost_metric,
                            # timeout=30.0,
                            # num_graph_seeds=14,
                            # num_trajopt_seeds=10,
                            # ik_fail_return=5,
                        ),
                    )
                    print("end motion gen: ",time.time() - start)
                else:
                    if total_spheres_tensor!= None: self.motion_gen.attach_spheres_to_robot(None, total_spheres_tensor, link_name="gripper")
                    motion_gen_result = self.motion_gen.plan_goalset(
                        start_state,
                        poses,
                        MotionGenPlanConfig(
                            max_attempts=self._CumotionActionServer__max_attempts,
                            enable_graph_attempt=1,
                            time_dilation_factor=time_dilation_factor,
                            pose_cost_metric=pose_cost_metric,
                        ),
                    )
                self.toggle_link_collision(plan_req.disable_collision_links, True)

            if motion_gen_result.success.item():
                result.error_code.val = MoveItErrorCodes.SUCCESS
                traj = self.get_joint_trajectory(
                    motion_gen_result.optimized_plan,
                    motion_gen_result.optimized_dt.item(),
                )
                result.planning_time = motion_gen_result.total_time
                result.planned_trajectory.append(traj)
                result.success = True
                if motion_gen_result.goalset_index != None:
                    result.goal_index = motion_gen_result.goalset_index.item()
            elif not motion_gen_result.valid_query:
                self.get_logger().error(
                    f"Invalid planning query: {motion_gen_result.status}"
                )
                if (
                    motion_gen_result.status
                    == MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS
                ):
                    result.error_code.val = MoveItErrorCodes.START_STATE_INVALID
                if motion_gen_result.status in [
                    MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION,
                    MotionGenStatus.INVALID_START_STATE_SELF_COLLISION,
                ]:
                    result.error_code.val = MoveItErrorCodes.START_STATE_IN_COLLISION
            else:
                self.get_logger().error(
                    f"Motion planning failed wih status: {motion_gen_result.status}"
                )
                if motion_gen_result.status == MotionGenStatus.IK_FAIL:
                    result.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
                if motion_gen_result.status in [MotionGenStatus.FINETUNE_TRAJOPT_FAIL, MotionGenStatus.TRAJOPT_FAIL, MotionGenStatus.GRAPH_FAIL]:
                    result.error_code.val = MoveItErrorCodes.TIMED_OUT

            self.get_logger().info(
                "returned planning result (query, success, failure_status): "
                + str(self._CumotionActionServer__query_count)
                + " "
                + str(motion_gen_result.success.item())
                + " "
                + str(motion_gen_result.status)
            )

        self._CumotionActionServer__query_count += 1
        result.message = "Planning completed with status: " + str(motion_gen_result.status)
        self.reset_planner()
        return result


def main(args=None):
    rclpy.init(args=args)
    cumotion_action_server = CumotionGoalSetPlannerServer()
    executor = MultiThreadedExecutor()
    executor.add_node(cumotion_action_server)
    try:
        executor.spin()
    except KeyboardInterrupt:
        cumotion_action_server.get_logger().info("KeyboardInterrupt, shutting down.\n")
    cumotion_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
