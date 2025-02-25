import mujoco
import numpy as np

# epsolon = 0.01            # m/(s^2) for "acceleration ~ 0"
# COLLISION_ELASTIC_FACTOR = 0.5
# FORCE_THRESHOLD = 10.0    # example for push/pull detection
# G = 9.8                   # gravity


class PhysicsTaxonomy:
    def __init__(self, objects):
        self.epsolon = 0.01  # m/(s^2) for "acceleration ~ 0"
        self.COLLISION_ELASTIC_FACTOR = 0.5
        self.FORCE_THRESHOLD = 10.0  # example for push/pull detection
        self.G = 9.8  # gravity
        self.precision = 3  # numbers after comma

        self.objects = objects

    def detect_linear_motion(self, vel_prev, vel_curr, dt):
        """
        check if the acceleration is 0 over the dt
        a = (v_current - v_prev)/dt
        a = dv(t)/dt
        """

        vel_curr = np.array(vel_curr)
        vel_prev = np.array(vel_prev)
        # Calculate acceleration
        acc = (vel_curr - vel_prev) / dt
        # the magnitude = sqrt(x^2 + y^2 + z^2)
        acc_norm = round(np.linalg.norm(acc), self.precision)
        labels = []
        if acc_norm < self.epsolon:
            labels.append("Constant Velocity")
        else:
            # sign of dot(vel, acc) can show speeding up or slowing down
            speed_change = np.dot(vel_curr, acc)
            if (
                abs(speed_change) < 1e-6
            ):  # Normalize small values to prevent misclassification
                speed_change = 0
            if speed_change > 0:
                labels.append("Accelerating")
            elif speed_change < 0:
                labels.append("Decelerating")

        if not len(labels):
            return None

        return {
            "category": "Kinematic Events",
            "subcategory": "Linear motion",
            "labels": labels,
        }

    def detect_projectile_motion(self):
        pass

    def detect_collision(self, model, data, prev_frame):
        """
        - check if the current object is in contact with other objects
        - get velocities before and after
        - calculate relative velocity before and after
        - calculate elasticity ratio = V_post/V_pre
        - if ratio > 0.5 : Elastic else InElastic
        """
        collision_results = {}

        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == 0 or g2 == 0:
                continue

            pair = tuple(sorted((g1, g2)))

            # compute normal direction from contact frame
            normal = c.frame[:3]

            # object IDs and velocities before the collision
            vel1_id = next(
                (obj["id"]
                 for obj in self.objects if obj["geom_id"] == g1), None
            )
            vel2_id = next(
                (obj["id"]
                 for obj in self.objects if obj["geom_id"] == g2), None
            )
            vel1_pre = prev_frame.get(vel1_id, {}).get("velocity", [])
            vel2_pre = prev_frame.get(vel2_id, {}).get("velocity", [])

            # joint addresses for post-collision velocities
            adr1 = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{g1}_free")]
            adr2 = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{g2}_free")]

            # post-collision velocities
            vel1_post = data.qvel[adr1: adr1 + 3].tolist()
            vel2_post = data.qvel[adr2: adr2 + 3].tolist()

            # relative velocity before and after
            rel_vel_pre = np.dot(
                (np.array(vel1_pre) - np.array(vel2_pre)), normal)
            rel_vel_post = np.dot(
                (np.array(vel1_post) - np.array(vel2_post)), normal)

            elasticity_ratio = (
                abs(rel_vel_post / rel_vel_pre) if rel_vel_pre != 0 else 0
            )

            collision_type = (
                "Elastic Collision"
                if elasticity_ratio > self.COLLISION_ELASTIC_FACTOR
                else "Inelastic Collision"
            )

            collision_results[pair] = {
                "category": "Interaction Events",
                "subcategory": "Collision",
                "labels": [collision_type],
            }

        return collision_results

    def get_taxonomy(self, model, data, dt, prev_frame):
        """
        loop through each of the object and extract labels
        returns {
            "object1": {

            }
        }
        """
        results = {}
        dt = max(dt, 1e-6)  # avoiding devision by 0

        collision_results = self.detect_collision(model, data, prev_frame)

        # for (g1, g2), collision_info in collision_results.items():
        #     obj1_id =list( filter(lambda x: x["geom_id"] == g1,  self.objects))[0]["id"]
        #     obj2_id =list( filter(lambda x: x["geom_id"] == g2,  self.objects))[0]["id"]

        #     results[obj1_id].append(collision_info)
        #     results[obj2_id].append(collision_info)

        for i, obj in enumerate(self.objects):
            object_id = obj.get("id", "")
            results[object_id] = []

            mass = obj.get("mass", 0)
            density = obj.get("density", 0)

            joint_name = f"obj{i}_free"
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            adr = model.jnt_dofadr[joint_id]

            velocity = data.qvel[adr: adr + 3].tolist()
            velocity = [round(v, 2) for v in velocity]
            angular_velocity = data.qvel[adr + 3: adr + 6].tolist()
            position = data.qpos[adr: adr + 3].tolist()

            vel_prev = prev_frame.get(object_id, {}).get("velocity", [])
            linear_motion = self.detect_linear_motion(
                vel_prev[:3], velocity, dt)
            if linear_motion:
                results[object_id].append(linear_motion)

        for (g1, g2), collision_info in collision_results.items():
            obj1_id = list(filter(lambda x: x["geom_id"] == g1, self.objects))[
                0]["id"]
            obj2_id = list(filter(lambda x: x["geom_id"] == g2, self.objects))[
                0]["id"]

            results[obj1_id].append(collision_info)
            results[obj2_id].append(collision_info)

        return results
