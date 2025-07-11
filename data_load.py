import os
import json
import re
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import matplotlib.colors as mcolors
from collections import defaultdict


class PHOLPhysicsDataset(Dataset):
    def __init__(
        self,
        root: str,
        video_transform=None,
        mask_transform=None,
        include_qa: bool = False,
    ):
        super().__init__()
        self.root = root
        all_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        valid = []
        for d in all_dirs:
            sd = os.path.join(root, d)
            if os.path.isfile(os.path.join(sd, "obj.json")) and \
               os.path.isfile(os.path.join(sd, "simulation_objects.mp4")):
                valid.append(d)

        
        numeric = sorted([d for d in valid if d.isdigit()], key=lambda x: int(x))
        non_numeric = sorted([d for d in valid if not d.isdigit()])
        self.scenes = numeric + non_numeric
        print("all_dirs", len(all_dirs))
        print("valid", len(valid))
        print("numeric", len(numeric))
        print("non_numeric", len(non_numeric))
        print(" self.scenes ", len( self.scenes ),  self.scenes[0])

        self.video_transform = video_transform or (lambda x: x)
        self.mask_transform = mask_transform or (lambda x: x)
        self.include_qa = include_qa
        self._collision_re = re.compile(r"collision", re.IGNORECASE)
        self._motion_re = re.compile(r"sliding|rolling|stationary|accelerating|decelerating", re.IGNORECASE)
        self._stationary_re = re.compile(r"stationary", re.IGNORECASE)

    def __len__(self):
        return len(self.scenes)

    def _load_json(self, path: str) -> Dict:
        with open(path, "r") as f:
            return json.load(f)

    def _get_physical_props(self, objects: List[Dict]) -> Dict:
        props = {}
        for obj in objects:
            fr = obj.get("friction", "").split()
            shape = obj.get("geom_type", "unknown")
            rgba_str = obj.get("visual", {}).get("rgba", "")
            color = [float(x) for x in rgba_str.split()] if rgba_str else []
            color_name = self.rgba_to_name(color)
            mass = obj.get("mass", None)
            elasticity = obj.get("elasticity", 0.0)
            velocity = obj.get("velocity", [0, 0, 0])
            try:
                mass = float(mass) if mass is not None else 1.0
            except (ValueError, TypeError):
                mass = 1.0
                
            props[obj["id"]] = {
                "mass": mass, 
                "friction": [float(x) for x in fr] if fr else [0.4],
                "elasticity": elasticity,
                "velocity": velocity,
                "position": [float(obj.get(f"init_possition_{ax}", 0)) for ax in ["x", "y"]] + [0],
                "material": obj.get("material", "unknown"),
                "shape": shape,
                "color": color_name,
            }
        return props

    def _compute_collision(self, frames: List[Dict]) -> bool:
        for fr in frames:
            if fr.get("interactions"):
                return True
            for o in fr.get("objects", {}).values():
                for tax in o.get("taxonomy", []):
                    if any(self._collision_re.search(lbl) for lbl in tax.get("labels", [])):
                        return True
        return False

    def _get_taxonomy(self, frames: List[Dict]) -> Dict[str, List[List[str]]]:
        taxonomy = {}
        for fr in frames:
            for obj_id, obj_state in fr.get("objects", {}).items():
                labels = []
                bbox = obj_state.get("bbox", [[0,0],[0,0]])
                if bbox and bbox != [[0, 0], [0, 0]]:
                    for tax_entry in obj_state.get("taxonomy", []):
                        labels.extend(tax_entry.get("labels", []))
                    taxonomy.setdefault(obj_id, []).append(labels)
        return taxonomy

    def _describe_obj(self, p={}):
        color_name = p.get("color", "unknown color")
        shape = p.get("shape", "object")
        material = p.get("material", "unknown material")
        return f"{color_name} {shape} made of {material}"

    def _has_stationary_objects(self, frames):
        for fr in frames:
            for obj_state in fr.get("objects", {}).values():
                for tax in obj_state.get("taxonomy", []):
                    if any(self._stationary_re.search(lbl) for lbl in tax.get("labels", [])):
                        return True
        return False

    def _get_heaviest_object(self, props):
        if not props:
            return "No objects"
        return max(props.items(), key=lambda x: x[1]["mass"])[0]

    def _identify_collision_pairs(self, annotations):
        collision_pairs = set()

        for frame in annotations.get("frames", []):
            for interaction in frame.get("interactions", []):
                involved_ids = [f"geom_obj{int(oid)-1}" for oid in interaction ]
                                # if f"geom_obj{int(oid)-1}" in valid_ids]
                if len(involved_ids) >= 2:
                    collision_pairs.add(tuple(sorted(involved_ids)))
                # collision_pairs.add(tuple(sorted(involved_ids)))
        
        output = [tuple(pair) for pair in collision_pairs]
        return output

    def _count_collision_objects(self, annotations):
        collided_objects = set()
        collision_pairs = self._identify_collision_pairs(annotations)
        for pair in collision_pairs:
            collided_objects.add(pair[0])
            collided_objects.add(pair[1])

        return len(collided_objects)

    def _get_most_collided_object(self, annotations):
        collision_counts = defaultdict(int)

        for frame in annotations.get("frames", []):
            for interaction in frame.get("interactions", []):
                for oid in interaction:
                    obj_id = f"geom_obj{int(oid)-1}"
                    collision_counts[obj_id] += 1

        
        # for fr in frames:
        #     for obj_id, obj_state in fr.get("objects", {}).items():
        #         for tax in obj_state.get("taxonomy", []):
        #             if any(self._collision_re.search(lbl) for lbl in tax.get("labels", [])):
                        # collision_counts[obj_id] += 1
        if not collision_counts:
            return "No collisions occurred"
        print(collision_counts)
        return max(collision_counts.items(), key=lambda x: x[1])[0]

    def _count_state_transitions(self, taxonomy):
        transitions = {
            'stopped_objects': set(),
            'moving_to_stationary': set(),
            'stationary_to_moving': set(),
            'rolling': set(),
        }

        # Rolling Motion
# Rolling Motion With Slipping
        
        object_behaviors = {}
        for obj_id, state_sequence in taxonomy.items():
            prev_state = None
            for states in state_sequence:
                current_state = states[-1] if states else None

                if current_state:
                    if current_state.lower() in ["rolling motion", "rolling motion with slipping"]:
                        transitions["rolling"].add(obj_id)
                
                # Check state transitions
                if prev_state and current_state:
                    prev_moving = any(t in prev_state.lower() 
                                    for t in ['moving', 'accelerating', 'decelerating'])
                    curr_stopped = any(t in current_state.lower() 
                                     for t in ['friction stop', 'stationary'])
                    
                    if prev_moving and curr_stopped:
                        # transitions['moving_to_stationary'][obj_id] += 1
                        transitions['moving_to_stationary'].add(obj_id)
                    
                    prev_stopped = any(t in prev_state.lower() 
                                     for t in ['friction stop', 'stationary'])
                    curr_moving = any(t in current_state.lower() 
                                    for t in ['accelerating'])

                    check_label = any(t in [current_state.lower()]  for t in ['stationary to moving'])
                    if prev_stopped and curr_moving and any(t in [prev_state.lower(), current_state.lower()]  for t in ['stationary to moving']):
                        # print("prev_state, current_state", prev_state, current_state)
                        # print("prev_stopped", prev_stopped)
                        # print("curr_moving", curr_moving)
                        # print("check_label", check_label)
                        # transitions['stationary_to_moving'][obj_id] += 1
                        transitions['stationary_to_moving'].add(obj_id)
                
                prev_state = current_state
            
            # Check if ever stopped
            if any('stopped' in s.lower() or 'stationary' in s.lower()
                      for states in state_sequence for s in states):
                    transitions['stopped_objects'].add(obj_id)
        
        result = {
            'stopped_objects': len(transitions['stopped_objects']),
            'moving_to_stationary': len(transitions['moving_to_stationary']),
            'stationary_to_moving': len(transitions['stationary_to_moving'])
        }
        return transitions

    def _generate_hypothetical_collision_questions(self, props, annotations, collision_pairs):
        questions = []
        
        for obj1_id, obj2_id in collision_pairs:
            p1, p2 = props.get(obj1_id), props.get(obj2_id)
            desc1 = self._describe_obj(p1)
            desc2 = self._describe_obj(p2)
            m1 = p1['mass']
            m2 = p2['mass']
            fric1 = p1['friction'][0] if p1['friction'] else 0.4
            fric2 = p2['friction'][0] if p2['friction'] else 0.4
            
            collision_frame_idx = None
            for i, frame in enumerate(annotations['frames']):
                if any("collision" in lbl.lower()
                      for obj in frame['objects'].values()
                      for tax in obj.get("taxonomy", [])
                      for lbl in tax.get("labels", [])):
                    collision_frame_idx = i
                    break
    
            if collision_frame_idx is None:
                continue
                
            pre_collision_frame = annotations['frames'][max(0, collision_frame_idx - 1)]
            pos1 = pre_collision_frame['objects'][obj1_id].get('position', [0,0,0])
            pos2 = pre_collision_frame['objects'][obj2_id].get('position', [0,0,0])
            vel1 = pre_collision_frame['objects'][obj1_id].get('velocity', [0,0,0])
            vel2 = pre_collision_frame['objects'][obj2_id].get('velocity', [0,0,0])

            def would_reach_collision(original_vel, original_mass, original_fric, modified_mass, modified_fri):
                # Friction deceleration: a = -μg
                original_decel = original_fric * 9.81
                modified_decel = modified_fri * 9.81
                
                # Time to stop with original parameters
                t_stop_original = sum(v**2 for v in original_vel)**0.5 / original_decel
                
                # Distance traveled until stop
                d_original = sum(v**2 for v in original_vel)**0.5 * t_stop_original - 0.5 * original_decel * t_stop_original**2
                
                # For modified parameters - velocity scales with sqrt(mass) for same energy
                modified_vel = [v * (modified_mass/original_mass)**0.5 for v in original_vel]
                t_stop_modified = sum(v**2 for v in modified_vel)**0.5 / modified_decel
                d_modified = sum(v**2 for v in modified_vel)**0.5 * t_stop_modified - 0.5 * modified_decel * t_stop_modified**2
                
                # Compare to actual distance to collision point
                distance_to_collision = sum((pos2[i]-pos1[i])**2 for i in range(3))**0.5
                return d_modified >= distance_to_collision

            would_collide_10x_mass = would_reach_collision(vel1, m1, fric1, m1*10, fric1)

            would_collide_high_friction = would_reach_collision(vel1, m1, fric1, m1, fric1*2)

            would_collide_both = would_reach_collision(vel1, m1, fric1, m1*10, fric1*2)
            questions.extend([
            {
                "question": f"If {desc1} was 10x heavier but with same friction, would it still collide with {desc2}?",
                "answer": "Yes" if would_collide_10x_mass else "No"
            }
            ])
        
            # # relative position and velocity
            # rel_pos = [pos2[i] - pos1[i] for i in range(3)]
            # rel_vel = [vel2[i] - vel1[i] for i in range(3)]
            
            # closing_speed = -sum(rel_pos[i] * rel_vel[i] for i in range(3)) / (sum(p**2 for p in rel_pos)**0.5)
            # time_to_collision = (sum(p**2 for p in rel_pos)**0.5) / max(closing_speed, 1e5)

            # moving_toward = time_to_collision > 0
            # if moving_toward:
            #     # Cross product gives distance between lines
            #     rel_motion_cross = [
            #         rel_pos[1] * rel_vel[2] - rel_pos[2]*rel_vel[1],
            #         rel_pos[2] * rel_vel[0] - rel_pos[0]*rel_vel[2],
            #         rel_pos[0] * rel_vel[1] - rel_pos[1]*rel_vel[0]
            #     ]
            #     min_distance = (sum(c**2 for c in rel_motion_cross)**0.5) / (sum(v**2 for v in rel_vel)**0.5)
            # else:
            #     min_distance = (sum(p**2 for p in rel_pos)**0.5)


            #  # Get object sizes (assuming radius-like property)
            # size1 = p1.get('size', 0.5)  # Default size if not specified
            # size2 = p2.get('size', 0.5)

            # would_collide = moving_toward and (min_distance < (size1 + size2))

            # questions.append({
            #         "question": f"In the collision between {desc1} and {desc2}, if {desc1} was 10x heavier, would they still collide?",
            #         "answer": "Yes" if would_collide else "No"
            #     })
            
            # Question about size change
            # original_distance = sum(p**2 for p in rel_pos)**0.5
            # size_increase_factor = 2  # Assume doubling size
            
            # # Check if increased size would cause collision
            # would_collide = original_distance < size_increase_factor * (p1.get('size', 1) + p2.get('size', 1))
            
            # questions.append({
            #     "question": f"If {desc1} was {size_increase_factor}x bigger in size, would it collide with {desc2}?",
            #     "answer": "Yes" if would_collide else "No",
    
            # })
            
            # # Combined mass and size question
            # questions.append({
            #     "question": f"If {desc1} was both 10x heavier and {size_increase_factor}x bigger, would it collide with {desc2}?",
            #     "answer": "Yes" if (moving_toward or would_collide) else "No",
            #     "explanation": (
            #         "Mass changes don't affect collision occurrence, but size does. " +
            #         ("The increased size would cause collision" if would_collide else 
            #          "Neither mass nor size increase would cause collision")
            #     ),
            #     "type": "hypothetical_combined_change"
            # })
        
        return questions

        
    def _make_qa_for_scene(self, scene: Dict) -> List[Dict[str, str]]:
        qas = []
        props = scene["physical_props"]
        taxonomy = scene["taxonomy"]
        has_collisions = scene["has_collisions"]
        annotations = scene.get("annotations", {})
        frames = annotations.get("frames", [])

        valid_ids = set(props.keys())

        #  property questions
        # for obj_id, p in props.items():
        #     desc = self._describe_obj(p)
        #     qas.extend([
        #         {
        #             "question": f"What is the mass of the {desc}?",
        #             "answer": f"{p['mass']:.2f} units" if isinstance(p['mass'], float) else str(p['mass'])
        #         },
        #         {
        #             "question": f"What is the friction coefficient of the {desc}?",
        #             "answer": f"{p['friction'][0]:.2f}" if p['friction'] else "unknown"
        #         },
        #         {
        #             "question": f"What material is the {desc} made of?",
        #             "answer": p['material']
        #         }, 
        #         {
        #             "question": f"How does the {p['shape']} shape of {desc} affect its motion?",
        #             "answer": f"The {p['shape']} shape affects motion by {'allowing rolling with less friction' if p['shape'] in ['sphere', 'cylinder'] else 'creating more surface contact and friction'}."
        #         }
        #     ])

        # scene questions
        qas.extend([
            {
                # "question": "Based on the video, do any objects come into contact or collide with each other during the scene?",
                "question": "Does the video show any instances where two or more objects make physical contact or collide with each other?",
                # "question": "Do any objects collide in this scene?",
                "answer": "Yes" if has_collisions else "No"
            },
            {
                # "question": "How many distinct physical objects appear in the scene?",
                "question": "What is the total count of visually distinct physical objects present at any point throughout the video scene?",
                # "question": "How many objects are in this scene?",
                "answer": str(len(props))
            }
        ])

        transitions = self._count_state_transitions(taxonomy)
        print("transitions", transitions)
        qas.extend([
            {
                "question": "How many objects come to a complete stop at any point during the video?",
                # "question": "How many objects have stopped in the video?",
                "answer": str(len(transitions['stopped_objects']))
            },
            {
                "question": "How many objects transition from moving to stationary states (i.e., decelerate to a stop) during the video?",
                # "question": "How many objects changing state from moving towards stationary (deacelerating) state during the video?",
                "answer": str(len(transitions['moving_to_stationary']))
            },
            {
                # "question": "Based on the video, how many distinct objects exhibit rolling motion at any point during the scene?",
                "question": "How many unique objects exhibit rotational motion characteristic of rolling at any point during the video observation?",
                "answer": str(len(transitions['rolling']))
                # "answer": f"{len(transitions["rolling"])}"
            }
            # {
            #     "question": "How many objects go from stationary to moving state in the video?",
            #     "answer": str(len(transitions['stationary_to_moving']))
            # }
        ])


        collision_pairs = self._identify_collision_pairs(annotations) if has_collisions else []
        qas.extend([
            {
                "question": "What is the total count of unique objects that participated in any collision event throughout the video?",
                # "question": "How many objects were involved in collisions?",
                "answer": str(self._count_collision_objects(annotations)) if has_collisions else "No collisions detected"
            },
            {
                "question": "Which object was involved in the highest number of unique collisions with other different objects during the video?",
                # "question": "Which object collided with the most other objects?",
                "answer": (
                    self._describe_obj(props[self._get_most_collided_object(annotations)]) 
                    if has_collisions else "No collisions detected"
                )
            },
            {
                "question": "What is the total number of separate collision events that occurred between different object pairs in the video?",
                # "question": "How many differen collitions have happened in the video?",
                "answer": str(len(collision_pairs))
            }
        ])
        

        
        # print("props", props)
        if has_collisions:
            # num_collided = self._count_collision_objects(frames)
            
            
            # qas.append({
            #     "question": "How many objects were involved in collisions?",
            #     "answer": str(num_collided)
            # })
            
            # most_collided = self._get_most_collided_object(annotations)
            # if most_collided in props:
            #     desc = self._describe_obj(props[most_collided])
            #     qas.append({
            #         "question": "Which object collided with the most other objects?",
            #         "answer": desc
            #     })
                
            collision_pairs = self._identify_collision_pairs(annotations)
            print("collision_pairs ", collision_pairs)
            # hypothetical_collision_questions = self._generate_hypothetical_collision_questions(props, annotations, collision_pairs)
            # qas.extend(hypothetical_collision_questions)

            kinomatic_qas = self._get_kinematic_loss(collision_pairs, props, annotations)
            qas.extend(kinomatic_qas)
            # for obj1_id, obj2_id in collision_pairs:
            #     p1, p2 = props.get(obj1_id), props.get(obj2_id)
            #     desc1 = self._describe_obj(p1)
            #     desc2 = self._describe_obj(p2)
            #     # 
            #     e1 = float(p1.get('elasticity', 0.7))
            #     e2 = float(p2.get('elasticity', 0.7))
            #     # coefficient of restitution (COR, or "elasticity")
            #     cor = (e1 + e2) / 2
            #     # Kinetic Energy Loss
            #     percent_ke_lost = (1 - cor ** 2) * 100

            #     m1, m2 = p1['mass'], p2['mass']
            #     if m1 > m2:
            #         more_mass = desc1
            #         less_mass = desc2
            #     else:
            #         more_mass = desc2
            #         less_mass = desc1

            #     qas.extend([
            #         {
            #             "question": f"What percentage of kinetic energy is lost when {desc1} collides with {desc2}?",
            #             "answer": (
            #                 f"{percent_ke_lost:.0f}%"
            #             )
            #         }
            #     ])


        # counterfactual_qas = self._generate_counterfactuals(props, taxonomy, collision, annotations)
        # qas.extend(counterfactual_qas)

        # motion_qas = self._generate_motion_questions(taxonomy, props)
        # qas.extend(motion_qas)

        # temporal_questions = self._generate_temporal_questions(props, annotations, valid_ids)
        # qas.extend(temporal_questions)
    

        return qas

    def _get_kinematic_loss(self, collision_pairs, props, annotations):
        questions = []
        print("_get_kinematic_loss collision_pairs", collision_pairs)
        for obj1_id, obj2_id in collision_pairs:
            p1, p2 = props.get(obj1_id, {}), props.get(obj2_id, {})
            desc1 = self._describe_obj(p1)
            desc2 = self._describe_obj(p2)

            collision_frames = []
            for i, frame in enumerate(annotations['frames']):
                obj1_state = frame['objects'].get(obj1_id, {})
                obj2_state = frame['objects'].get(obj2_id, {})
                
                # Check if collision is happening in this frame
                collision_detected = False
                for tax in obj1_state.get('taxonomy', []):
                    if any("collision" in lbl.lower() for lbl in tax.get('labels', [])):
                        if obj2_id in [oid for oid in frame['objects'] if oid != obj1_id]:
                            collision_detected = True
                            break
                
                if collision_detected:
                    collision_frames.append(i)
            
            # print("collision_frames", collision_frames)
            if not collision_frames:
                continue

            # peak collision frame (max velocity change)
            peak_frame_idx = collision_frames[0]
            max_delta_v = 0
            for i in collision_frames:
                frame = annotations['frames'][i]
                v1 = frame['objects'].get(obj1_id, {}).get('velocity', [0,0,0])
                v2 = frame['objects'].get(obj2_id, {}).get('velocity', [0,0,0])
                delta_v = sum((v1[i]-v2[i])**2 for i in range(3)) 
                if delta_v > max_delta_v:
                    max_delta_v = delta_v
                    peak_frame_idx = i

            # 
            collision_start = min(collision_frames)
            collision_end = max(collision_frames)
            
            # Get states before, during, and after collision
            pre_frame = annotations['frames'][max(0, peak_frame_idx - 1)]
            post_frame = annotations['frames'][min(len(annotations['frames']) - 1, peak_frame_idx + 1)]
            # pre_frame = annotations['frames'][max(0, collision_start - 1)]
            # post_frame = annotations['frames'][min(len(annotations['frames']) - 1, collision_end + 1)]
            
            # Get velocities
            v1_before = pre_frame['objects'][obj1_id].get('velocity', [0,0,0])
            v2_before = pre_frame['objects'][obj2_id].get('velocity', [0,0,0])
            v1_after = post_frame['objects'][obj1_id].get('velocity', [0,0,0])
            v2_after = post_frame['objects'][obj2_id].get('velocity', [0,0,0])

            m1, m2 = p1['mass'], p2['mass']
    
            def kinetic_energy(v, m):
                return 0.5 * m * sum(vi**2 for vi in v)

            # Individual object losses
            ke1_before = kinetic_energy(v1_before, m1)
            ke1_after = kinetic_energy(v1_after, m1)
            percent_ke1_lost = 0
            if ke1_before > 0:
                percent_ke1_lost = max(0, 100 * (ke1_before - ke1_after) / ke1_before)
            # percent_ke1_lost = max(0, 100 * (ke1_before - ke1_after) / ke1_before) if ke1_before > 0 else 0
    
            ke2_before = kinetic_energy(v2_before, m2)
            ke2_after = kinetic_energy(v2_after, m2)
            percent_ke2_lost = 0
            if ke2_before > 0:
                percent_ke2_lost = max(0, 100 * (ke2_before - ke2_after) / ke2_before)

            # percent_ke2_lost = max(0, 100 * (ke2_before - ke2_after) / ke2_before) if ke2_before > 0 else 0


            # System loss 
            KE_before = ke1_before + ke2_before
            KE_after = ke1_after + ke2_after
            percent_ke_lost = max(0, 100 * (KE_before - KE_after) / KE_before) if KE_before > 0 else 0
            # KE_before = kinetic_energy(v1_before, m1) + kinetic_energy(v2_before, m2)
            # KE_after = kinetic_energy(v1_after, m1) + kinetic_energy(v2_after, m2)

            # print('KE_before ', KE_before)
            # if KE_before > 0:
            #     percent_ke_lost = max(0, 100 * (KE_before - KE_after) / KE_before)  # Don't allow negative loss
            # else:
            #     percent_ke_lost = 0
            
            # calculate actual coeff of restitution - COR
            v_rel_before = sum((v1i - v2i)**2 for v1i, v2i in zip(v1_before, v2_before))**0.5
            v_rel_after = sum((v1i - v2i)**2 for v1i, v2i in zip(v1_after, v2_after))**0.5
            
            if v_rel_before > 0:
                actual_cor = min(max(v_rel_after / v_rel_before, 0), 1.0)  # Bound between 0 and 1
            else:
                actual_cor = 0
            
            #  collision duration
            frame_rate = 25  
            collision_duration = (collision_end - collision_start + 1) / frame_rate

            questions.extend([
                {
                    "question": f"What is total kinetic energy loss of the system when {desc1} collides with {desc2}?",
                    # "answer": f"{percent_ke_lost:.1f}%",
                    "answer": f"{percent_ke_lost:.1f}%",
                    "details": {
                        "system_loss": f"{percent_ke_lost:.1f}%",
                        f"{obj1_id}_loss": f"{percent_ke1_lost:.1f}%",
                        f"{obj2_id}_loss": f"{percent_ke2_lost:.1f}%"
                    }
                }, 
                {
                    "question": f"How much kinetic energy did {desc1} lose in the collision?",
                    "answer": f"{percent_ke1_lost:.1f}%"
                },
                {
                    "question": f"How much kinetic energy did {desc2} lose in the collision?",
                    "answer": f"{percent_ke2_lost:.1f}%"
                },
                {
                    "question": f"How long did the collision between {desc1} and {desc2} last?",
                    "answer": f"{collision_duration:.2f} seconds"
                }
            ])

        # print('questions', questions)
        return questions
            

    def _calculate_trajectories(self, annotations, valid_ids):
        trajectories = {}
        for obj_id in valid_ids:
            trajectories[obj_id] = {
                'positions': [],
                'velocities': [],
                'first_move_frame': float('inf'),
                'last_move_frame': -1,
                'max_distance': 0
            }
        
        for frame_idx, frame in enumerate(annotations.get("frames", [])):
            for obj_id, obj_state in frame.get("objects", {}).items():
                if obj_id in valid_ids:
                    pos = obj_state.get("position", [0,0,0])
                    vel = obj_state.get("velocity", [0,0,0])
                    trajectories[obj_id]['positions'].append(pos)
                    trajectories[obj_id]['velocities'].append(vel)
                    
                    if sum(abs(v) for v in vel) > 0.1:  # If moving
                        trajectories[obj_id]['first_move_frame'] = min(
                            trajectories[obj_id]['first_move_frame'], frame_idx)
                        trajectories[obj_id]['last_move_frame'] = max(
                            trajectories[obj_id]['last_move_frame'], frame_idx)
        
        for obj_id in trajectories:
            if len(trajectories[obj_id]['positions']) > 1:
                start = trajectories[obj_id]['positions'][0]
                end = trajectories[obj_id]['positions'][-1]
                trajectories[obj_id]['max_distance'] = sum(
                    (e-s)**2 for s,e in zip(start, end))**0.5
    
        return trajectories
        

    def rgba_to_name(self, rgba):
        if not rgba or len(rgba) < 3:
            return "unknown color"
        
        rgb = tuple(rgba[:3])
        min_dist = float('inf')
        best_name = "unknown color"
        
        for name, hex_val in mcolors.CSS4_COLORS.items():
            named_rgb = mcolors.to_rgb(hex_val)
            dist = sum((c1 - c2)**2 for c1, c2 in zip(rgb, named_rgb))
            if dist < min_dist:
                min_dist = dist
                best_name = name
                
        return best_name.replace('grey', 'gray').replace('gray', 'grey')  # Standardize spelling

        
    def __getitem__(self, idx: int) -> Dict:
        sid = self.scenes[idx]
        scene_dir = os.path.join(self.root, sid)
        d = self._load_json(os.path.join(scene_dir, "obj.json"))
        frames = d.get("frames", [])

        for frame in frames:
            objects = frame.get("objects", {})
            # create new dict without empty bbox objects
            frame["objects"] = {
                obj_id: obj_state 
                for obj_id, obj_state in objects.items()
                if obj_state.get("bbox", [[0,0],[0,0]]) != [[0, 0], [0, 0]]
            }

        valid_ids = {
            obj_id
            for fr in frames
            for obj_id, obj_state in fr.get("objects", {}).items()
            if obj_state.get("bbox", [[0,0],[0,0]]) != [[0, 0], [0, 0]]
        }

        physical_props = self._get_physical_props(d.get("objects", []))
        physical_props = {oid: physical_props[oid] for oid in valid_ids if oid in physical_props}

        has_collisions = self._compute_collision(frames)
        taxonomy = self._get_taxonomy(frames)

        vid = self.video_transform(
            self._read_video(os.path.join(scene_dir, "simulation_objects.mp4")))
        seg = self.mask_transform(
            self._read_video(os.path.join(scene_dir, "simulation_objects_segmentation.mp4")))

        qa = None
        if self.include_qa:
            scene = {
                "physical_props": physical_props,
                "taxonomy": taxonomy,
                "has_collisions": has_collisions,
                "annotations": d,
            }
            qa = self._make_qa_for_scene(scene)

        return {
            "scene_id": sid,
            "video": vid,
            "segmentation": seg,
            "physical_props": physical_props,
            "has_collisions": has_collisions,
            "taxonomy": taxonomy,
            "annotations": d,
            "qa": qa,
        }

    def _read_video(self, path: str) -> torch.Tensor:
        vr = VideoReader(path, ctx=cpu(0))
        frames = vr.get_batch(range(len(vr))).asnumpy()
        return torch.from_numpy(frames).permute(3, 0, 1, 2).float().div(255.0)