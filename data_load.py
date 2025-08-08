import os
import json
import re
import random
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
        fps: int = 25
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
        print(" self.scenes ", len(self.scenes),  self.scenes[0])

        self.fps = fps
        self.video_transform = video_transform or (lambda x: x)
        self.mask_transform = mask_transform or (lambda x: x)
        self.include_qa = include_qa
        self._collision_re = re.compile(r"collision", re.IGNORECASE)
        self._motion_re = re.compile(r"sliding|rolling|stationary|accelerating|decelerating", re.IGNORECASE)
        self._stationary_re = re.compile(r"stationary", re.IGNORECASE)
        self.MIN_VISIBLE_SIZE = 10

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
                bbox = obj_state.get("bbox", [[0, 0], [0, 0]])
                if bbox and bbox != [[0, 0], [0, 0]]:
                    for tax_entry in obj_state.get("taxonomy", []):
                        labels.extend(tax_entry.get("labels", []))
                    taxonomy.setdefault(obj_id, []).append(labels)
        return taxonomy

    def _describe_obj(self, p={}):
        color_name = p.get("color", "unknown color")
        shape = p.get("shape", "object")
        material = p.get("material", "unknown material")
        return f"{color_name} {shape}"
        # made of {material}"

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
                involved_ids = [f"geom_obj{int(oid)-1}" for oid in interaction]
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
            objects = frame.get("objects", {})
            for interaction in frame.get("interactions", []):
                for oid in interaction:
                    # geom_obj1
                    obj_id = f"geom_obj{int(oid)-1}"
                    obj_state = objects.get(obj_id)
                    if not obj_state:
                        continue
                    bbox = obj_state.get("bbox", [[0, 0], [0, 0]])
                    if bbox and bbox != [[0, 0], [0, 0]]:
                        collision_counts[obj_id] += 1
                    # collision_counts[obj_id] += 1

        # for fr in frames:
        #     for obj_id, obj_state in fr.get("objects", {}).items():
        #         for tax in obj_state.get("taxonomy", []):
        #             if any(self._collision_re.search(lbl) for lbl in tax.get("labels", [])):
                        # collision_counts[obj_id] += 1

        if not collision_counts:
            return None

        max_count = max(collision_counts.values())
        most_collided = [obj_id for obj_id, count in collision_counts.items()
                         if count == max_count]
        # print("most_collided", most_collided)

        return most_collided
        # if not collision_counts:
        #     return "No collisions occurred"
        # return max(collision_counts.items(), key=lambda x: x[1])[0]

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

                    check_label = any(t in [current_state.lower()] for t in ['stationary to moving'])
                    if prev_stopped and curr_moving and any(t in [prev_state.lower(), current_state.lower()] for t in ['stationary to moving']):
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

    def _make_qa_for_scene(self, scene: Dict) -> List[Dict[str, str]]:
        fps = 25
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
                # "question": "Does the video show any instances where two or more objects make physical contact or collide with each other?",
                # "question": "Do any objects collide in this scene?",
                "question": "Are there any moments in the video where two or more objects collide or make physical contact?",
                "answer": "Yes" if has_collisions else "No"
            },
            {
                # "question": "How many distinct physical objects appear in the scene?",
                # "question": "What is the total count of visually distinct physical objects present at any point throughout the video scene?",
                "question": "How many distinct physical objects appear during the video?",
                # "question": "How many objects are in this scene?",
                "answer": str(len(props))
            }
        ])

        transitions = self._count_state_transitions(taxonomy)
        # print("transitions", transitions)
        qas.extend([
            {
                "question": "How many objects come to a complete stop during the video that we can see?",
                # "question": "How many objects have stopped in the video?",
                "answer": str(len(transitions['stopped_objects']))
            },
            # {
            #     "question": "How many objects slow down and eventually stop during the video?",
            #     # "question": "How many objects transition from moving to stationary states (i.e., decelerate to a stop) during the video?",
            #     # "question": "How many objects changing state from moving towards stationary (deacelerating) state during the video?",
            #     "answer": str(len(transitions['moving_to_stationary']))
            # },
            {
                # "question": "Based on the video, how many distinct objects exhibit rolling motion at any point during the scene?",
                "question": "How many objects display rolling motion at any point in the video?",
                # "question": "How many unique objects exhibit rotational motion characteristic of rolling at any point during the video observation?",
                "answer": str(len(transitions['rolling']))
                # "answer": f"{len(transitions["rolling"])}"
            }
            # {
            #     "question": "How many objects go from stationary to moving state in the video?",
            #     "answer": str(len(transitions['stationary_to_moving']))
            # }
        ])

        all_pairs = self._identify_collision_pairs(annotations) if has_collisions else []
        collision_pairs = [
            (a, b) for (a, b) in all_pairs
            if a in props and b in props
        ]

        # 2) Build the QAs
        qas.extend([
            {
                # "question": "What is the total count of unique objects that participated in any collision event throughout the video?",
                "question": "How many unique objects were involved in collision throughout the video?",
                "answer": (
                    str(self._count_collision_objects(annotations))
                    if has_collisions else "No collisions detected"
                )
            },
            # {
            #     "question": "Which object was involved in the most collisions with other objects?",
            #     # "question": "Which object was involved in the highest number of unique collisions with other different objects during the video?",
            #     "answer": (
            #         # get the most‐collided object ID, verify it’s in props, then describe
            #         self._describe_obj(props[self._get_most_collided_object(annotations)])
            #         if has_collisions and self._get_most_collided_object(annotations) in props
            #         else "No collisions detected"
            #     )
            # },
            # {
            #     "question": "How many distinct collision events occurred between object pairs in the video?",
            #     # "question": "What is the total number of separate collision events that occurred between different object pairs in the video?",
            #     "answer": str(len(collision_pairs))
            # }
        ])

        most_collided = self._get_most_collided_object(annotations)
        props = scene["physical_props"]
        if has_collisions and most_collided:
            # Get all visible objects for options
            all_visible_objects = list(props.keys())
            options = [self._describe_obj(props[obj_id]) for obj_id in all_visible_objects]

            # Get correct answers (only the most collided objects)
            correct_answers = [self._describe_obj(props[obj_id]) for obj_id in most_collided]

            # Shuffle options but keep track of correct ones
            random.shuffle(options)

            qas.append({
                "question": "Which object was involved in the most collisions with other objects?",
                "options": options,
                "answer": correct_answers,
                "multiple_answers": len(correct_answers) > 1
            })
        else:
            qas.append({
                "question": "Which object was involved in the most collisions with other objects?",
                "answer": "No collisions detected"
            })

        # collision_pairs = self._identify_collision_pairs(annotations) if has_collisions else []
        # qas.extend([
        #     {
        #         "question": "What is the total count of unique objects that participated in any collision event throughout the video?",
        #         # "question": "How many objects were involved in collisions?",
        #         "answer": str(self._count_collision_objects(annotations)) if has_collisions else "No collisions detected"
        #     },
        #     {
        #         "question": "Which object was involved in the highest number of unique collisions with other different objects during the video?",
        #         # "question": "Which object collided with the most other objects?",
        #         "answer": (
        #             self._describe_obj(props[self._get_most_collided_object(annotations)])
        #             if has_collisions else "No collisions detected"
        #         )
        #     },
        #     {
        #         "question": "What is the total number of separate collision events that occurred between different object pairs in the video?",
        #         # "question": "How many differen collitions have happened in the video?",
        #         "answer": str(len(collision_pairs))
        #     }
        # ])

        # print("props", props)
        if has_collisions:
            collision_pairs = self._identify_collision_pairs(annotations)
            # print("collision_pairs ", collision_pairs)

            kinomatic_qas = self._get_kinematic_loss(collision_pairs, props, annotations)
            qas.extend(kinomatic_qas)

        stationary_re = self._stationary_re
        for obj_id, p in props.items():
            # count frames where this object is labelled stationary
            count_stat = 0
            for fr in frames:
                obj = fr.get("objects", {}).get(obj_id)
                if not obj:
                    continue
                for tax in obj.get("taxonomy", []):
                    if any(stationary_re.search(lbl) for lbl in tax.get("labels", [])):
                        count_stat += 1
                        break
            true_seconds = round(count_stat / self.fps, 2)
            #
            cands = {
                round(true_seconds, 2),
                round(max(true_seconds * 0.8, 0), 2),
                round(true_seconds * 1.2, 2),
                round(abs(true_seconds - 1.0), 2)
            }

            # If rounding collapsed any values, add new offsets until we have 4
            delta = 0.25          # seconds
            while len(cands) < 4:
                cands.add(round(true_seconds + delta, 2))
                delta += 0.25

            opts = [f"{v:.2f}s" for v in cands]
            random.shuffle(opts)  # in-place

            # opts = [
            #     f"{true_seconds:.2f}s",
            #     f"{max(true_seconds * 0.8, 0):.2f}s",
            #     f"{true_seconds * 1.2:.2f}s",
            #     f"{abs(true_seconds - 1.0):.2f}s"
            # ]
            # random.shuffle(opts)

            qas.append({
                "question": f"How many seconds did the {self._describe_obj(p)} spend stationary?",
                "options": opts,
                "answer": f"{true_seconds:.2f}s",
                "explanation": (
                    f"Count the number of video frames labelled ‘stationary’ for this object "
                    f"(and divide by the frame-rate ({self.fps} fps)"
                )
            })

            first_stat_frame = None
            for i, fr in enumerate(frames):
                obj_state = fr.get("objects", {}).get(obj_id)
                if not obj_state:
                    continue
                if any(self._stationary_re.search(lbl)
                       for tax in obj_state.get("taxonomy", [])
                       for lbl in tax.get("labels", [])):
                    first_stat_frame = i
                    break

            # print("first_stat_frame ", first_stat_frame)
            if first_stat_frame is not None:
                start_time = round(first_stat_frame / self.fps, 2)   # seconds

                # ---- build 4 unique options ----
                cand_ts = {
                    round(start_time, 2),
                    round(max(start_time - 0.4, 0), 2),     # a bit earlier
                    round(start_time + 0.4, 2),             # a bit later
                    round(abs(start_time - 1.0), 2)         # unrelated offset
                }
                # top-up if rounding caused duplicates
                delta = 0.25
                while len(cand_ts) < 4:
                    cand_ts.add(round(start_time + delta, 2))
                    delta += 0.25

                time_opts = [f"{v:.2f}s" for v in cand_ts]
                random.shuffle(time_opts)

                qas.append({
                    "question": (
                        f"At what time in the video does the {self._describe_obj(p)} "
                        f"first become stationary?"),
                    "options": time_opts,
                    "answer": f"{start_time:.2f}s",
                    "explanation": (
                        "Scan frames in order until the first one labelled ‘stationary’ "
                        f"is found .  Divide that frame number by the frame-rate ({self.fps} fps) "
                    )
                    # "explanation": (
                    #     "Scan frames in order until the first one labelled ‘stationary’ "
                    #     f"is found (frame {first_stat_frame}).  Divide that frame number "
                    #     f"by the frame-rate ({self.fps} fps) → {first_stat_frame} ÷ "
                    #     f"{self.fps} = {start_time:.2f} s."
                    # )
                })

        options = []
        for obj_id, p in props.items():
            desc = self._describe_obj(p)
            options.append(desc)
        if props:
            max_fr_obj = max(props.items(), key=lambda kv: kv[1]["friction"][0])[0]
            correct = self._describe_obj(props[max_fr_obj])

            options = [self._describe_obj(p) for p in props.values()]
            qas.append({
                "question": "Which object had the highest friction coefficient?",
                "options": options,
                "answer": correct
            })
        else:
            print("Warning: No valid physical properties found. Skipping friction question.")
        # find max friction
        # max_fr_obj = max(props.items(), key=lambda kv: kv[1]["friction"][0])[0]
        # correct = self._describe_obj(props[max_fr_obj])

        # # append as multiple-choice
        # qas.append({
        #     "question": "Which object had the highest friction coefficient?",
        #     "options": options,
        #     "answer": correct
        # })
        return qas

    def _get_kinematic_loss(self, collision_pairs, props, annotations):
        questions = []

        for obj1_id, obj2_id in collision_pairs:
            # Skip if either object is not in props
            if obj1_id not in props or obj2_id not in props:
                continue

            p1, p2 = props[obj1_id], props[obj2_id]
            desc1 = self._describe_obj(p1)
            desc2 = self._describe_obj(p2)

            # Find all frames where both objects exist
            valid_frames = [
                (i, frame) for i, frame in enumerate(annotations['frames'])
                if obj1_id in frame['objects'] and obj2_id in frame['objects']
            ]

            if not valid_frames:
                continue

            # Find peak collision frame (max velocity change)
            peak_frame_idx = 0
            max_delta_v = 0
            for i, frame in valid_frames:
                v1 = frame['objects'][obj1_id].get('velocity', [0, 0, 0])
                v2 = frame['objects'][obj2_id].get('velocity', [0, 0, 0])
                delta_v = sum((v1[i]-v2[i])**2 for i in range(3))
                if delta_v > max_delta_v:
                    max_delta_v = delta_v
                    peak_frame_idx = i

            # Get frames before and after collision
            pre_frame_idx = max(0, peak_frame_idx - 1)
            post_frame_idx = min(len(annotations['frames']) - 1, peak_frame_idx + 1)

            # Get pre and post frames - must check if objects exist in these frames
            pre_frame = annotations['frames'][pre_frame_idx]
            post_frame = annotations['frames'][post_frame_idx]

            # Skip if either object is missing in pre or post frame
            if (obj1_id not in pre_frame['objects'] or obj2_id not in pre_frame['objects'] or
                    obj1_id not in post_frame['objects'] or obj2_id not in post_frame['objects']):
                continue

            # Get velocities
            v1_before = pre_frame['objects'][obj1_id].get('velocity', [0, 0, 0])
            v2_before = pre_frame['objects'][obj2_id].get('velocity', [0, 0, 0])
            v1_after = post_frame['objects'][obj1_id].get('velocity', [0, 0, 0])
            v2_after = post_frame['objects'][obj2_id].get('velocity', [0, 0, 0])

            m1, m2 = p1['mass'], p2['mass']

            def kinetic_energy(v, m):
                return 0.5 * m * sum(vi**2 for vi in v)

            ke1_before = kinetic_energy(v1_before, m1)
            ke1_after = kinetic_energy(v1_after, m1)
            percent_ke1_lost = max(0, 100 * (ke1_before - ke1_after) / ke1_before) if ke1_before > 0 else 0

            ke2_before = kinetic_energy(v2_before, m2)
            ke2_after = kinetic_energy(v2_after, m2)
            percent_ke2_lost = max(0, 100 * (ke2_before - ke2_after) / ke2_before) if ke2_before > 0 else 0

            KE_before = ke1_before + ke2_before
            KE_after = ke1_after + ke2_after
            percent_ke_lost = max(0, 100 * (KE_before - KE_after) / KE_before) if KE_before > 0 else 0

            # collision_duration = (valid_frames[-1][0] - valid_frames[0][0] + 1) / self.fps  # Assuming 25fps
            contact_frames = []
            for i, frame in enumerate(annotations['frames']):
                for interaction in frame.get('interactions', []):
                    mapped = {f"geom_obj{int(oid)-1}" for oid in interaction}
                    if {obj1_id, obj2_id}.issubset(mapped):
                        contact_frames.append(i)
                        break
            if contact_frames:
                start, end = min(contact_frames), max(contact_frames)
                collision_duration = (end - start + 1) / self.fps
            else:
                collision_duration = 0.0

            def make_opts(true_val, is_percent=True):
                a = round(true_val * 0.8, 1 if is_percent else 2)
                b = round(min(true_val + (10 if is_percent else 0.2),
                          100 if is_percent else true_val + 1), 1 if is_percent else 2)
                c = round(abs(true_val - (50 if is_percent else 0.5)), 1 if is_percent else 2)
                opts = list({true_val, a, b, c})
                random.shuffle(opts)
                suffix = "%" if is_percent else "s"
                return [f"{v:.1f}{suffix}" if is_percent else f"{v:.2f}{suffix}" for v in opts]

            true_dur = round(collision_duration, 2)

            # Distractors: 80%, 120%, and a fixed offset
            # opts_dur = [
            #     f"{true_dur:.2f}s",
            #     f"{true_dur * 0.8:.2f}s",
            #     f"{true_dur * 1.2:.2f}s",
            #     f"{abs(true_dur - 0.5):.2f}s"
            # ]
            opts_dur = list({
                f"{true_dur:.2f}s",
                f"{true_dur * 0.8:.2f}s",
                f"{true_dur * 1.2:.2f}s",
                f"{abs(true_dur - 0.5):.2f}s"
            })
            random.shuffle(opts_dur)

            true_val = round(percent_ke_lost, 1)
            # opts = [
            #     f"{true_val:.1f}%",
            #     f"{max(true_val - 20, 0):.1f}%",   # e.g. 60% if true=80%
            #     f"{min(true_val + 10, 100):.1f}%", # e.g. 90% if true=80%
            #     f"{abs(true_val - 50):.1f}%"
            # ]
            opts = list({
                f"{true_val:.1f}%",
                f"{max(true_val - 20, 0):.1f}%",
                f"{min(true_val + 10, 100):.1f}%",
                f"{abs(true_val - 50):.1f}%"
            })
            random.shuffle(opts)
            questions.extend([
                {
                    # "question": f"What is total kinetic energy loss of the system when {desc1} collides with {desc2}?",
                    "question": f"What percentage of the system’s kinetic energy was lost when the {desc1} collided with the {desc2}?",
                    "answer": f"{true_val:.1f}%",
                    "options":  make_opts(round(true_val, 1)),
                    "explanation": ("For each object, kinetic energy KE = 0.5·m·|v|². "
                                    "Compute KE_before and KE_after just before and just after impact, "
                                    "then %loss = 100·(KE_before − KE_after)/KE_before. "
                                    "Sum the two objects to get system % loss."),
                    "details": {
                        "system_loss": f"{percent_ke_lost:.1f}%",
                        f"{obj1_id}_loss": f"{percent_ke1_lost:.1f}%",
                        f"{obj2_id}_loss": f"{percent_ke2_lost:.1f}%"
                    }
                },
                # {
                #     "question": f"How much kinetic energy did {desc1} lose during the collision?",
                #     "answer": f"{percent_ke1_lost:.1f}%"
                # },
                # {
                #     "question": f"How much kinetic energy did {desc2} lose during the collision?",
                #     "answer": f"{percent_ke2_lost:.1f}%"
                # },
                {
                    "question": f"How long did the collision between {desc1} and {desc2} last (video fps is {self.fps})?",
                    "options": make_opts(true_dur, is_percent=False),
                    "answer": f"{true_dur:.2f}s",
                    "explanation": ("Count consecutive frames where the interaction list "
                                    "contains both objects, then divide by fps "
                                    f"({self.fps}).")
                }
                # {
                #     "question": f"How long did the collision between {desc1} and {desc2} last?",
                #     "answer": f"{collision_duration:.2f} seconds"
                # }
            ])

        return questions

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

    def is_visibly_valid(self, bbox):
        if bbox == [[0, 0], [0, 0]]:
            return False
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        return width >= self.MIN_VISIBLE_SIZE and height >= self.MIN_VISIBLE_SIZE

    def __getitem__(self, idx: int) -> Dict:
        sid = self.scenes[idx]
        print('sid', sid)
        scene_dir = os.path.join(self.root, sid)
        d = self._load_json(os.path.join(scene_dir, "obj.json"))
        frames = d.get("frames", [])

        for frame in frames:
            objects = frame.get("objects", {})
            # create new dict without empty bbox objects
            frame["objects"] = {
                obj_id: obj_state
                for obj_id, obj_state in objects.items()
                # if obj_state.get("bbox", [[0,0],[0,0]]) != [[0, 0], [0, 0]]
                if self.is_visibly_valid(obj_state.get("bbox", [[0, 0], [0, 0]]))
            }

        valid_ids = {
            obj_id
            for fr in frames
            for obj_id, obj_state in fr.get("objects", {}).items()
            # if obj_state.get("bbox", [[0,0],[0,0]]) != [[0, 0], [0, 0]]
            if self.is_visibly_valid(obj_state.get("bbox", [[0, 0], [0, 0]]))
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
