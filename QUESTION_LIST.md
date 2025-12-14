# Question List - Grouped by Topic

This document lists all questions generated from `phlop/advanced_physics_questions.py` and `phlop/question_answer.py`, organized by topic/category.

---

## 1. Counting

### Object Count
- **Question**: "How many distinct physical objects appear in the video?"
- **Source**: `question_answer.py` → `get_questions_answers()`
- **Type**: Numerical
- **Difficulty**: Easy
- **Question Type**: `object_count`

---

## 2. Motion Analysis

### Rolling Motion Detection
- **Question**: "How many objects exhibit rolling motion at any point?"
- **Source**: `question_answer.py` → `get_questions_answers()`
- **Type**: Numerical
- **Difficulty**: Easy
- **Question Type**: `rolling_detection`

### Stopped Objects Count
- **Question**: "How many objects come to a complete stop during the video that we can see?"
- **Source**: `question_answer.py` → `get_questions_answers()`
- **Type**: Numerical
- **Difficulty**: Easy
- **Question Type**: `stopped_objects_count`

### Stationary Duration
- **Question**: "How many seconds did the {object} spend stationary?"
- **Source**: `question_answer.py` → `get_questions_answers()`
- **Type**: Multiple Choice
- **Difficulty**: Medium
- **Question Type**: `stationary_duration`

### Stationary Start Time
- **Question**: "At what time in the video does the {object} first become stationary?"
- **Source**: `question_answer.py` → `get_questions_answers()`
- **Type**: Multiple Choice
- **Difficulty**: Medium
- **Question Type**: `stationary_start_time`

---

## 3. Collision Detection

### Collision Presence
- **Question**: "Are there any collisions between objects in the video?"
- **Source**: `question_answer.py` → `get_questions_answers()`
- **Type**: Yes/No
- **Difficulty**: Easy
- **Question Type**: `collision_presence`

---

## 4. Collision Physics

### Momentum Conservation
- **Question**: "During the collision at t={time}s, the momentum ratio is {ratio}. What does this indicate?"
- **Source**: `question_answer.py` → `_collision_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Hard
- **Question Type**: `momentum_conservation`

---

## 5. Collision Geometry

### Relative Velocity Magnitude
- **Question**: "At t={time}s, what was the relative velocity magnitude between {object1} and {object2} just before collision?"
- **Source**: `advanced_physics_questions.py` → `generate_collision_geometry_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Medium
- **Question Type**: `relative_velocity_magnitude`
- **Category**: Collision Geometry

### Relative Velocity Decision (30% of cases)
- **Question**: "At t={time}s, was the relative velocity between {object1} and {object2} high enough (above 1.5 m/s) to cause a highly inelastic collision?"
- **Source**: `advanced_physics_questions.py` → `generate_collision_geometry_questions()`
- **Type**: Multiple Choice (Decision Question)
- **Difficulty**: Medium
- **Question Type**: `relative_velocity_decision`
- **Category**: Collision Geometry
- **Note**: 30% of relative velocity questions are converted to decision questions with threshold reasoning

---

## 6. Post-Collision Motion

### Direction Reversal
- **Question**: "At t={time}s, after the collision, did {object} reverse its direction of motion?"
- **Source**: `advanced_physics_questions.py` → `generate_post_collision_motion_questions()`
- **Type**: Yes/No
- **Difficulty**: Medium
- **Question Type**: `direction_reversal`
- **Category**: Post-Collision Motion

---

## 7. Mass & Density

### Mass Ratio
- **Question**: "What is the mass ratio between {heaviest_object} (heaviest) and {lightest_object} (lightest)?"
- **Source**: `advanced_physics_questions.py` → `generate_mass_effects_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Medium
- **Question Type**: `mass_ratio`
- **Category**: Mass & Density

---

## 8. Material Properties / Physical Properties

### Friction Coefficient Comparison (Advanced)
- **Question**: "Between {object1} and {object2}, which has a higher friction coefficient, and by approximately how much?"
- **Source**: `advanced_physics_questions.py` → `generate_friction_coefficient_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Medium
- **Question Type**: `friction_coefficient_comparison`
- **Category**: Material Properties

### Highest Friction Coefficient (Basic)
- **Question**: "Which object had the highest friction coefficient?"
- **Source**: `question_answer.py` → `get_questions_answers()`
- **Type**: Multiple Choice
- **Difficulty**: Medium
- **Question Type**: `friction_comparison`
- **Category**: Physical Properties

---

## 9. Geometry & Shape

### Shape Distribution
- **Question**: "What is the distribution of object shapes in the simulation?"
- **Source**: `advanced_physics_questions.py` → `generate_shape_distribution_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Easy
- **Question Type**: `shape_distribution`
- **Category**: Geometry & Shape

---

## 10. Comparative Questions

### Fastest Object (Peak Velocity)
- **Question**: "Which object reached the highest peak velocity during the simulation?"
- **Source**: `advanced_physics_questions.py` → `generate_velocity_comparison_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Medium
- **Question Type**: `fastest_object`
- **Category**: Comparative Questions

---

## 11. Counterfactual Reasoning

### Velocity Scaling (Quadratic Relationship)
- **Question**: "At t={time}s, {object} has velocity {speed} m/s. If the initial velocity doubled, how much farther would it slide (assuming same friction)?"
- **Source**: `advanced_physics_questions.py` → `generate_velocity_scaling_counterfactual_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Hard
- **Question Type**: `velocity_scaling`
- **Category**: Counterfactual Reasoning

### Friction Scaling
- **Question**: "A {object} slides for {duration}s before stopping. If the friction coefficient were doubled, what would most likely happen?"
- **Source**: `advanced_physics_questions.py` → `generate_counterfactual_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Hard
- **Question Type**: `friction_scaling`
- **Category**: Counterfactual Reasoning

---

## 12. Property Competition

### Property Competition (Mass vs Friction)
- **Question**: "After colliding, {object1} (mass: {mass1} kg, friction: {friction1}) traveled farther than {object2} (mass: {mass2} kg, friction: {friction2}), despite being lighter/having higher friction. Why did this happen?"
- **Source**: `advanced_physics_questions.py` → `generate_property_competition_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Very Hard
- **Question Type**: `property_competition`
- **Category**: Property Competition
- **Note**: Tests understanding of how conflicting properties (mass, friction) interact to determine outcomes

---

## 13. Conceptual Physics

### Newton's Second Law
- **Question**: "According to Newton's Second Law (F = m*a), if an object's mass doubles but the applied force remains the same, what happens to its acceleration?"
- **Source**: `advanced_physics_questions.py` → `generate_physics_principle_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Medium
- **Question Type**: `newtons_second_law`
- **Category**: Conceptual Physics

### Stationary but Rotating (Apparent Contradiction)
- **Question**: "At t={time}s, a {object} is stationary but rotating. Is this physically possible?"
- **Source**: `advanced_physics_questions.py` → `generate_contradictory_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Medium
- **Question Type**: `apparent_contradiction`
- **Category**: Conceptual Physics

### Rolling with Slipping (Apparent Contradiction)
- **Question**: "At t={time}s, a {object} is both rolling and sliding. Can an object do both at the same time?"
- **Source**: `advanced_physics_questions.py` → `generate_contradictory_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Medium
- **Question Type**: `apparent_contradiction`
- **Category**: Conceptual Physics

### Temporal Consistency (Stationary → Accelerating)
- **Question**: "At t={time1}s, {object} is labeled as 'Stationary'. At t={time2}s, it is labeled as 'Accelerating'. Is this transition consistent?"
- **Source**: `advanced_physics_questions.py` → `generate_contradictory_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Hard
- **Question Type**: `temporal_consistency`
- **Category**: Conceptual Physics
- **Note**: For val/test splits, uses velocity inference instead of labels (split-aware masking)

### Label vs Observation Mismatch
- **Question**: "At t={time}s, {object} is labeled as 'Accelerating', but its velocity magnitude decreased from {prev_speed} m/s to {curr_speed} m/s. Is this possible?"
- **Source**: `advanced_physics_questions.py` → `generate_contradictory_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Hard
- **Question Type**: `label_observation_mismatch`
- **Category**: Conceptual Physics
- **Note**: Tests understanding that acceleration is a vector; speed can decrease while accelerating

---

## 14. Temporal Reasoning

### Event Sequence
- **Question**: "What is the correct chronological order of the following events: {event1}, {event2}, ..."
- **Source**: `advanced_physics_questions.py` → `generate_temporal_sequence_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Medium
- **Question Type**: `event_sequence`
- **Category**: Temporal Reasoning

---

## 15. Causal Reasoning

### Direct Causation (Motion Change)
- **Question**: "At t={time}s, {object1} collides with {object2}. The second object was stationary before the collision but moves afterward. Is the collision the cause?"
- **Source**: `advanced_physics_questions.py` → `generate_causal_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Hard
- **Question Type**: `direct_causation`
- **Category**: Causal Reasoning

### Energy Analysis (Collision Type)
- **Question**: "At t={time}s, {object1} collides with {object2}. The collision is classified as '{collision_type}'. What does this tell us about energy conservation?"
- **Source**: `advanced_physics_questions.py` → `generate_causal_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Hard
- **Question Type**: `energy_analysis`
- **Category**: Causal Reasoning
- **Note**: For val/test splits, collision type is inferred from energy loss instead of using taxonomy labels (split-aware masking)

### Energy Analysis (Inferred from Energy Loss - Val/Test Splits)
- **Question**: "At t={time}s, {object1} collides with {object2}. Based on the observed energy loss ({ke_loss}%), what can we infer about energy conservation?"
- **Source**: `advanced_physics_questions.py` → `generate_causal_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Hard
- **Question Type**: `energy_analysis`
- **Category**: Causal Reasoning
- **Note**: Only generated for val/test splits. Uses thresholds matching `physics_engine.py`: <10% = Elastic, 10-50% = Partially Inelastic, ≥50% = Highly Inelastic

---

## 16. Energy Analysis

### Kinetic Energy Loss Percentage
- **Question**: "What percentage of the system's kinetic energy was lost when the {object1} collided with the {object2}?"
- **Source**: `advanced_physics_questions.py` → `generate_causal_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Very Hard
- **Question Type**: `kinetic_energy_loss`
- **Category**: Energy Analysis

### Kinetic Energy Loss Decision (30% of cases)
- **Question**: "When {object1} collided with {object2}, was the kinetic energy loss significant enough (≥50%) to classify this as a highly inelastic collision?"
- **Source**: `advanced_physics_questions.py` → `generate_causal_questions()`
- **Type**: Multiple Choice (Decision Question)
- **Difficulty**: Very Hard
- **Question Type**: `kinetic_energy_loss_decision`
- **Category**: Energy Analysis
- **Note**: 30% of energy loss questions are converted to decision questions. Threshold (50%) matches `physics_engine.py` (ke_ratio ≤ 0.5)

---

## 17. Multi-Hop Reasoning

### Indirect Causation (3-hop chain: A → B → C)
- **Question**: "{object1} hits {object2}. Later, {object2} hits {object3}. Is {object1} indirectly responsible for the second collision?"
- **Source**: `advanced_physics_questions.py` → `generate_multihop_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Very Hard
- **Question Type**: `indirect_causation`
- **Category**: Multi-Hop Reasoning

### Four-Hop Causation (A → B → C → D)
- **Question**: "{object1} hits {object2}. Then {object2} hits {object3}. Finally, {object3} hits {object4}. Is {object1} indirectly responsible for the collision between {object3} and {object4}?"
- **Source**: `advanced_physics_questions.py` → `generate_multihop_questions()`
- **Type**: Multiple Choice
- **Difficulty**: Very Hard
- **Question Type**: `four_hop_causation`
- **Category**: Multi-Hop Reasoning
- **Note**: System first attempts to find 4-hop chains, falls back to 3-hop if not found

---

## Summary Statistics

### By Source File

**From `question_answer.py`:**
- Total question types: 8
- Categories: Counting, Motion Analysis, Collision Detection, Collision Physics, Physical Properties
- Questions: Object Count, Rolling Motion Detection, Stopped Objects Count, Stationary Duration, Stationary Start Time, Collision Presence, Momentum Conservation, Highest Friction Coefficient

**From `advanced_physics_questions.py`:**
- Total question types: 22
- Categories: Collision Geometry, Post-Collision Motion, Mass & Density, Material Properties, Geometry & Shape, Comparative Questions, Counterfactual Reasoning, Property Competition, Conceptual Physics, Temporal Reasoning, Causal Reasoning, Energy Analysis, Multi-Hop Reasoning
- Questions: Relative Velocity Magnitude, Relative Velocity Decision, Direction Reversal, Mass Ratio, Friction Coefficient Comparison, Shape Distribution, Fastest Object, Velocity Scaling, Friction Scaling, Property Competition, Newton's Second Law, Stationary but Rotating, Rolling with Slipping, Temporal Consistency, Label vs Observation Mismatch, Event Sequence, Direct Causation, Energy Analysis (with split-aware masking), Kinetic Energy Loss, Kinetic Energy Loss Decision, Indirect Causation (3-hop), Four-Hop Causation

**Total**: 30 question types

### By Difficulty Level

- **Easy**: 5 question types
  - Object Count, Rolling Motion Detection, Stopped Objects Count, Collision Presence, Shape Distribution
- **Medium**: 13 question types
  - Stationary Duration, Stationary Start Time, Relative Velocity Magnitude, Relative Velocity Decision, Direction Reversal, Mass Ratio, Friction Coefficient Comparison (Advanced), Highest Friction Coefficient (Basic), Fastest Object, Newton's Second Law, Stationary but Rotating, Rolling with Slipping, Event Sequence
- **Hard**: 7 question types
  - Momentum Conservation, Velocity Scaling, Friction Scaling, Direct Causation, Energy Analysis (Collision Type), Temporal Consistency, Label vs Observation Mismatch
- **Very Hard**: 5 question types
  - Kinetic Energy Loss Percentage, Kinetic Energy Loss Decision, Indirect Causation (3-hop), Four-Hop Causation, Property Competition

**Total**: 30 question types

### By Answer Type

- **Numerical**: 3 question types
  - Object Count, Rolling Motion Detection, Stopped Objects Count
- **Yes/No**: 2 question types
  - Collision Presence, Direction Reversal
- **Multiple Choice**: 25 question types
  - All remaining questions (including new decision questions)

---

## Notes

1. Some questions use dynamic placeholders (e.g., `{object}`, `{time}`, `{speed}`) that are filled in at runtime based on simulation data.

2. Questions are generated conditionally based on:
   - Presence of specific events (collisions, motion types, etc.)
   - Number of objects in the simulation
   - Available physics data (velocities, masses, friction coefficients, etc.)

3. Both files include deduplication logic to avoid generating identical questions.

4. Questions are shuffled randomly before being returned.

5. **Split-Aware Label Masking**: For val/test splits, taxonomy labels are masked and questions require inference from raw velocity/position data. This ensures fair evaluation without ground truth hints.

6. **Decision Questions**: 30% of numeric questions (relative velocity, energy loss) are converted to decision questions with threshold reasoning, improving reasoning depth.

7. **Threshold Matching**: All energy loss thresholds match `physics_engine.py`:
   - Elastic: <10% loss (ke_ratio > 0.9)
   - Partially Inelastic: 10-50% loss (ke_ratio > 0.5)
   - Highly Inelastic: ≥50% loss (ke_ratio ≤ 0.5)

8. **Code Optimization**: The implementation uses caching for object descriptions, properties, and peak velocities to avoid repeated calculations.

5. **Question Generation Logic for Multiple Elements:**
   
   The system uses different strategies when multiple elements (objects, collisions, events) are available:
   
   **A. Single Comparison Question (Pick Extremes):**
   - **Friction Coefficient** (`generate_friction_coefficient_questions`): Collects all objects with friction, sorts them, picks the **highest and lowest**, generates **ONE** comparison question between them.
   - **Mass Ratio** (`generate_mass_effects_questions`): Collects all objects, sorts by mass, picks **heaviest and lightest**, generates **ONE** ratio question.
   - **Fastest Object** (`generate_velocity_comparison_questions`): Finds peak velocity for all objects, picks **fastest**, generates **ONE** question.
   - **Shape Distribution** (`generate_shape_distribution_questions`): Counts all shapes, generates **ONE** question about the overall distribution.
   
   **B. One Question Per Method (Early Return):**
   - **Collision Geometry** (`generate_collision_geometry_questions`): Loops through collisions, generates **ONE** question for the first valid collision, then **returns immediately** (line 191-192).
   - **Post-Collision Motion** (`generate_post_collision_motion_questions`): Loops through collisions, generates **ONE** question for the first valid collision, then **returns immediately** (line 254-255).
   - **Velocity Scaling** (`generate_velocity_scaling_counterfactual_questions`): Loops through objects, generates **ONE** question for the first object meeting criteria (speed > 1.0 and eventually stops), then **returns immediately** (line 525-526).
   
   **C. Multiple Questions Per Unique Collision (Up to 3):**
   - **Causal Questions** (`generate_causal_questions`): Uses a `seen_collisions` set to track processed collisions. For each **unique collision**, can generate up to **3 questions**:
     - Q1: Direct causation (if object goes from stationary to moving)
     - Q2: Energy analysis (based on collision type)
     - Q3: Kinetic energy loss percentage
   - Each collision is only processed once (deduplicated by time and object pair).
   
   **D. One Question Per Object/Event (Multiple Possible):**
   - **Counterfactual Friction** (`generate_counterfactual_questions`): Loops through **all objects**, generates **one question per object** that has a sliding event (sliding → friction stop sequence).
   - **Contradictory Questions** (`generate_contradictory_questions`): Uses boolean flags to ensure only **one question per contradiction type** (e.g., one for "stationary but rotating", one for "rolling with slipping").
   
   **E. Limited to First N Collisions:**
   - **Momentum Conservation** (`question_answer.py` → `_collision_questions`): Processes only the **first 3 collisions** (`collisions[:3]`), generates one question per collision if momentum data is available.
   
   **F. All Objects as Options:**
   - **Highest Friction** (`question_answer.py`): Finds the object with maximum friction, but includes **ALL objects** as options in the multiple choice question (line 330).
   
   **G. Property Competition Questions:**
   - **Property Competition** (`generate_property_competition_questions`): Analyzes collisions to find scenarios where properties conflict (e.g., lighter object travels farther). Generates **ONE** question per qualifying collision scenario, then returns immediately.
   
   **Summary:**
   - Most comparison-based questions generate **ONE** question by selecting extremes (highest/lowest, fastest, etc.).
   - Event-based questions (collisions, motion) typically generate **ONE** question per method with early return.
   - Causal questions are the exception: they can generate **multiple questions per collision** (up to 3), but each collision is processed only once.
   - Counterfactual questions can generate **multiple questions** (one per object/event that meets criteria).
   - Property competition questions generate **ONE** question per qualifying collision scenario.
