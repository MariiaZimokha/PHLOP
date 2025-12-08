# PHLOP 
**P**hysics-grounded **H**ierarchical **L**atent space for **O**bject state and **P**resentation.  

PHLOP is an open dataset and benchmark designed to assess the physical
reasoning capabilities of video–language models. It provides
synthetic videos generated via the MuJoCo physics engine, along with
multi‑modal annotations (RGB frames, segmentation masks, bounding boxes and
per–frame taxonomy labels) and automatically generated question–answer pairs.
These questions probe a wide range of Newtonian concepts—collision dynamics,
rolling and sliding, energy dissipation, friction and more—and can be
presented to models under different prompting conditions (video only,
physics parameters, taxonomy labels or both).



---

![PHLOP](assets/kikis-delivery-service-tired.gif)

| Level 1 (General Category)    | Level 2 (Event Type)        | Level 3 (Specific Event Labels)               |
|------------------------------- |---------------------------- |----------------------------------------------- |
| **Kinematic Events**          | Linear Motion               | Constant Velocity                             |
|                               |                             | Decelerating                                  |
|                               |                             | Accelerating                                  |
|                               |                             | Stationary                                    |
|                               | ~~Projectile Motion~~           | ~~With Air Resistance~~                           |
|                               |                             | ~~Without Air Resistance~~                       |
|                               | Rotational Motion           | Pure Spinning                                 |
|                               |                             | Rolling Motion                                |
|                               |                             | Rolling Motion with Slipping                                |
|                               |                             | Spinning While Sliding                  |
| **Interaction Events**        | Collisions                  | Elastic Collision                             |
|                               |                             | Inelastic Collision                           |
|                               | ~~Rebounds~~                    | ~~Surface Rebound~~                               |
|                               |                             | ~~Momentum Transfer~~                             |
| **State Transitions**         | Motion Change               | Moving to Stopping                            |
|                               |                             | Stationary to Moving                            |
|                               | ~~External Force Effect~~       | ~~Object Stopped by Friction,~~                   |
|                               |                             | ~~Object Moved by External Force~~                |
| **Environmental Interactions**| Friction-Induced Events     | Friction Stop                                 |
|                               |                             | Sliding with Friction                         |
|                               |                             | ~~Drag Force Effects~~                            |


### Getting started
#### Prerequisites
PHLOP relies on the [MuJoCo](https://github.com/google-deepmind/mujoco)
physics engine for simulation and [PyTorch](https://pytorch.org/) for
model inference.  Ensure you have a recent Python (3.9+) and install
dependencies with:

```bash
pip install -r requirements.txt
```


#### Generating the dataset
To reproduce the PHLOP dataset, run the provided generation script:

```bash
python main.py --output_dir generated/ --num_videos 10
```
This script uses controlled randomisation over object shapes, material properties, initial velocities and global scene parameters (camera, lighting, floor) to produce diverse scenes. It will write videos and associated JSON logs to the specified output directory.

### Physics

# Physics Taxonomy Equations Reference

## 1. Kinematic Events

### 1.1 Linear Motion

#### Stationary
Object at rest with negligible velocity.

**Physics Definition:**
$$|v| < v_{\text{threshold}} \approx 0.001 \text{ m/s}$$

**Properties:**
- Zero or near-zero velocity
- Zero or near-zero acceleration
- Static friction can act (if on incline or with external forces)

---

#### Constant Velocity
Object moving at fixed speed in same direction.

**Physics Definition:**
$$|a| = \left|\frac{dv}{dt}\right| < a_{\text{threshold}} \approx 0.001 \text{ m/s}^2$$

$$v_{\text{current}} = v_{\text{previous}} + O(\text{threshold})$$

**Properties:**
- Net force is zero or negligible
- Velocity magnitude remains constant
- No external acceleration

---

#### Accelerating
Object increasing speed in its direction of motion.

**Physics Definition:**
$$|v_{\text{current}}| > |v_{\text{previous}}|$$

$$a = \frac{\Delta v}{\Delta t} > 0 \text{ (in direction of motion)}$$

**Properties:**
- Positive acceleration along velocity direction
- Often due to external force application
- Kinetic energy increasing

---

#### Decelerating
Object decreasing speed; motion opposes acceleration.

**Physics Definition:**
$$|v_{\text{current}}| < |v_{\text{previous}}|$$

$$a = \frac{\Delta v}{\Delta t} < 0 \text{ (opposes velocity)}$$

$$\vec{v} \cdot \vec{a} < 0$$

**Physics Sources:**
- Friction: $a = -\mu_k g$
- Air resistance: $a = -\frac{1}{2}\rho C_D A v^2 / m$
- Collision impulse

**Common Cause:** Kinetic friction

**Stopping Distance:**
$$s = \frac{v_0^2}{2\mu_k g}$$

**Stopping Time:**
$$t_{\text{stop}} = \frac{v_0}{\mu_k g}$$

---

### 1.2 Rotational Motion

#### Pure Spinning
Object rotating about its center with no translation.

**Physics Definition:**
$$|v| < v_{\text{threshold}} \quad \text{AND} \quad |\omega| > \omega_{\text{min}} \approx 0.5 \text{ rad/s}$$

**Angular Velocity:**
$$\vec{\omega} = \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}$$

**Properties:**
- Center of mass stationary
- Non-zero angular velocity
- Rotation happens about center
- No rolling or sliding motion

**Examples:**
- Spinning coin on table
- Gyroscope in fixed location
- Top spinning in place

---

#### Rolling Motion
Smooth rolling without slipping (pure rolling).

**Physics Definition (All must be true):**

1. **Rotation axis perpendicular to velocity:**
$$|\vec{\omega} \cdot \vec{v}| / (|\vec{\omega}| |\vec{v}|) < 0.3$$

2. **No-slip condition (rolling constraint):**
$$v = r \omega$$

where $r$ is the radius of curvature.

**Contact Point Velocity:**
$$\vec{v}_{\text{contact}} = \vec{v}_{\text{center}} - r\vec{\omega} = 0$$

**Kinetic Energy:**
$$KE = \frac{1}{2}mv^2 + \frac{1}{2}I\omega^2$$

For solid sphere: $I = \frac{2}{5}mr^2$
$$KE = \frac{1}{2}mv^2 + \frac{1}{2}\cdot\frac{2}{5}mr^2\omega^2 = \frac{1}{2}mv^2(1 + 0.4) = 0.7mv^2$$

**Shape Restriction:**
Only smooth curved surfaces can roll:
- ✓ Sphere
- ✓ Ball
- ✓ Cylinder
- ✗ Cube (tumbles instead)
- ✗ Block (tumbles instead)

---

#### Rolling Motion with Slipping
Rolling with slip; contact point has non-zero velocity.

**Physics Definition (All must be true):**

1. **Rotation axis perpendicular to velocity:**
$$|\vec{\omega} \cdot \vec{v}| / (|\vec{\omega}| |\vec{v}|) < 0.3$$

2. **Slip condition (violates rolling constraint):**
$$v \neq r\omega$$

**Contact Point Velocity (slipping velocity):**
$$v_{\text{slip}} = v_{\text{center}} - r\omega \neq 0$$

**Friction During Slip:**
Kinetic friction acts at contact: $f = \mu_k N$

**Coefficient of Restitution affects slipping:**
$$e = -\frac{\text{relative velocity after}}{\text{relative velocity before}}$$

**Energy Loss:**
$$\Delta KE = KE_{\text{after}} - KE_{\text{before}} < 0$$

Energy dissipated as heat: $Q = f \cdot v_{\text{slip}} \cdot t$

**Common Cause:** Skidding wheel on ice, sliding ball with rotation

**Note:** Can eventually transition to pure rolling if friction reduces slip speed.

---

#### Spinning While Sliding
Rotation axis parallel or oblique to velocity; no rolling relationship.

**Physics Definition (Any of these):**

1. **Axis NOT perpendicular to velocity:**
$$|\vec{\omega} \cdot \vec{v}| / (|\vec{\omega}| |\vec{v}|) > 0.3$$

2. **Linear and angular motion independent:**
$$\text{No relationship between } v \text{ and } \omega$$

3. **Contact point continuously changing:**
Different part of object contacts ground each instant

**Axis Alignment Definition:**
$$\text{axis\_alignment} = \frac{|\vec{\omega} \cdot \vec{v}|}{|\vec{\omega}| |\vec{v}|}$$

- $\approx 0$: Perpendicular (rolling possible) ← Rolling Motion
- $\approx 1$: Parallel (no rolling) ← Spinning While Sliding
- $0 < x < 1$: Oblique (no rolling) ← Spinning While Sliding

**Typical Examples:**
- Cube sliding horizontally while spinning about vertical axis
- Coin sliding while spinning on its face
- Cylinder sliding sideways with axial spin

**Shape Note:** All shapes fall into this category when:
$$\text{axis\_alignment} > 0.3$$

---

## 2. Environmental Interactions

### 2.1 Friction-Induced Events

#### Sliding with Friction
Object moving with kinetic friction causing deceleration.

**Physics Definition:**

**During Motion Phase** (v > v_threshold):
$$v > 0 \quad \text{AND} \quad a \approx -\mu_k g$$

**Kinetic Friction Equation:**
$$f_k = \mu_k N = \mu_k mg \text{ (on horizontal surface)}$$

**Acceleration from friction:**
$$a = -\mu_k g$$

**Magnitude Tolerance** (realistic simulation noise):
$$0.8 \leq \frac{|a|}{\mu_k g} \leq 1.2$$

**Velocity as function of time:**
$$v(t) = v_0 - \mu_k g \cdot t$$

**Stopping time:**
$$t_{\text{stop}} = \frac{v_0}{\mu_k g}$$

**Stopping distance:**
$$s = v_0 t_{\text{stop}} - \frac{1}{2}\mu_k g t_{\text{stop}}^2 = \frac{v_0^2}{2\mu_k g}$$

**Work done by friction:**
$$W_f = -f_k \cdot s = -\mu_k mg \cdot \frac{v_0^2}{2\mu_k g} = -\frac{1}{2}mv_0^2$$

All kinetic energy converted to heat.

**Deceleration Phase Detection:**
$$\vec{v} \cdot \vec{a} < 0 \quad \text{(opposite directions)}$$

---

#### Friction Stop
Object has stopped due to friction; now at rest.

**Physics Definition:**

**Final State** (v ≈ 0):
$$|v| < v_{\text{threshold}} \approx 0.001 \text{ m/s}$$

$$|a| \approx 0$$

**Transition Point:**
From "Sliding with Friction" to "Friction Stop":
$$v(t_{\text{stop}}) = 0$$

**Static Friction now active:**
$$|f_s| \leq \mu_s N$$

If on incline:
$$\mu_s \geq \tan(\theta) \quad \text{for object to remain at rest}$$

**Note on Semantics:**
- "Friction Stop" is the **result** of friction (state: at rest)
- "Sliding with Friction" is the **process** (state: moving with friction)
- Both involve friction but occur at different times
- Object transitions: Moving → Sliding with Friction → Friction Stop

**Temporal Sequence:**
```
Time t:     Sliding with Friction (v > 0, a = -μ·g)
Time t_stop: Transition point (v → 0)
Time t > t_stop: Friction Stop (v = 0, static friction holds)
```

---

## 3. State Transitions

### 3.1 Motion Change

#### Moving to Stopping
Transition from dynamic motion to rest state.

**Physics Definition:**
$$v_{\text{before}} > v_{\text{threshold}} \quad \text{AND} \quad v_{\text{after}} < v_{\text{threshold}}$$

**Detection Criteria:**
$$\text{prev\_motion} \in [\text{Constant Velocity}, \text{Accelerating}, \text{Decelerating}]$$
$$\text{curr\_motion} = \text{Stationary}$$

**Possible Causes:**
1. **Friction deceleration:** $a = -\mu_k g$
2. **Collision momentum loss:** Inelastic collision
3. **Obstacle impact:** Contact with barrier
4. **Combined:** Multiple friction sources

**Kinetic Energy at Stopping:**
$$KE_{\text{final}} = \frac{1}{2}m v_{\text{threshold}}^2 \approx 0$$

**If due to friction alone:**
$$v_0^2 = 2\mu_k g s$$

---

#### Stationary to Moving
Transition from rest to dynamic motion.

**Physics Definition:**
$$v_{\text{before}} < v_{\text{threshold}} \quad \text{AND} \quad v_{\text{after}} > v_{\text{threshold}}$$

**Detection Criteria:**
$$\text{prev\_motion} = \text{Stationary}$$
$$\text{curr\_motion} \in [\text{Constant Velocity}, \text{Accelerating}]$$

**Possible Causes:**
1. **Collision impulse:** Object hit by another
2. **External force:** Applied force overcomes static friction
3. **Incline motion:** Gravity overcomes static friction ($mg\sin\theta > \mu_s mg\cos\theta$)
4. **Surface change:** Moves to low-friction surface

**Force Override Condition (overcoming static friction):**
$$F_{\text{applied}} > \mu_s N$$

**Acceleration from external force:**
$$a = \frac{F_{\text{net}}}{m} = \frac{F_{\text{applied}} - f_s}{m}$$

---

## 4. Interaction Events

### 4.1 Collisions

#### Elastic Collision
Objects collide with kinetic energy conserved (bouncing).

**Physics Definition:**

**Coefficient of Restitution:**
$$e = -\frac{v_{\text{sep}}}{v_{\text{app}}} = -\frac{(v_1' - v_2') \cdot \hat{n}}{(v_1 - v_2) \cdot \hat{n}}$$

where $\hat{n}$ is the contact normal direction.

**Elastic Condition:**
$$e \geq 0.4 \quad \text{(or adjustable threshold)}$$

**Energy Conservation:**
$$KE_{\text{before}} = KE_{\text{after}}$$

$$\frac{1}{2}m_1 v_1^2 + \frac{1}{2}m_2 v_2^2 = \frac{1}{2}m_1 v_1'^2 + \frac{1}{2}m_2 v_2'^2$$

**Momentum Conservation (always true):**
$$m_1 \vec{v}_1 + m_2 \vec{v}_2 = m_1 \vec{v}_1' + m_2 \vec{v}_2'$$

**1D Case (head-on):**
$$v_1' = \frac{(m_1 - m_2)v_1 + 2m_2 v_2}{m_1 + m_2}$$

$$v_2' = \frac{(m_2 - m_1)v_2 + 2m_1 v_1}{m_1 + m_2}$$

**Special Cases:**
- Same mass: velocities exchange
- $m_2 >> m_1$: light object bounces back
- $m_1 = m_2$, $v_2 = 0$: first object stops, second moves

---

#### Inelastic Collision
Objects collide with kinetic energy dissipated (absorbing impact).

**Physics Definition:**

**Coefficient of Restitution:**
$$e < 0.4$$

**Energy Conservation (violated):**
$$KE_{\text{after}} < KE_{\text{before}}$$

**Energy Loss:**
$$\Delta E = KE_{\text{before}} - KE_{\text{after}} = Q > 0$$

Dissipated as heat, sound, deformation.

**Momentum Conservation (always true):**
$$m_1 \vec{v}_1 + m_2 \vec{v}_2 = m_1 \vec{v}_1' + m_2 \vec{v}_2'$$

**Perfectly Inelastic** (e = 0):
Objects stick together:
$$v' = \frac{m_1 v_1 + m_2 v_2}{m_1 + m_2}$$

**Energy loss in perfectly inelastic:**
$$\Delta E = \frac{1}{2}\frac{m_1 m_2}{m_1 + m_2}(v_1 - v_2)^2$$

**Partial Inelasticity** (0 < e < 1):
Objects separate but lose energy
$$KE_{\text{after}} = e^2 \cdot KE_{\text{before}}$$

---

## 5. Physics Constants Used

| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| Gravity | $g$ | 9.81 | m/s² |
| Velocity Threshold | $v_t$ | 0.001 | m/s |
| Acceleration Threshold | $a_t$ | 0.001 | m/s² |
| Rotation Threshold | $\omega_t$ | 0.5 | rad/s |
| Axis Alignment Threshold | - | 0.3 | (unitless) |
| Friction Tolerance | - | ±20% | (relative) |
| Elastic Threshold | $e_t$ | 0.4 | (unitless) |
| Rolling Epsilon | $\epsilon$ | 0.01 | m/s |

---

## 6. Summary: Label Assignment Logic

### Kinematic Events → Linear Motion

```
Compute: v_current = ||velocity||

IF v_current < v_threshold:
    → STATIONARY
ELSE:
    Compute: a = dv/dt
    
    IF ||a|| < a_threshold:
        → CONSTANT VELOCITY
    ELSE IF v_current > v_previous:
        → ACCELERATING
    ELSE:
        → DECELERATING
```

### Kinematic Events → Rotational Motion

```
Compute: v = ||velocity||, ω = ||angular_velocity||

IF v < v_threshold AND ω < 0.5:
    → NO ROTATION
ELSE IF v < v_threshold AND ω > 0.5:
    → PURE SPINNING
ELSE IF v > v_threshold AND ω < 0.1:
    → NO ROTATION (linear only)
ELSE IF v > v_threshold AND ω > 0.1:
    Compute: axis_alignment = |ω̂ · v̂|
    
    IF axis_alignment > 0.3:
        → SPINNING WHILE SLIDING
    ELSE:  (axis perpendicular, rolling possible)
        Compute: v_expected = r·ω
        Compute: rolling_diff = |v - v_expected|
        
        IF rolling_diff < ε:
            → ROLLING MOTION
        ELSE:
            → ROLLING MOTION WITH SLIPPING
```

### Environmental Interactions → Friction-Induced Events

```
IF v < v_threshold:
    → NO FRICTION EVENT (object at rest)
ELSE:
    Compute: a = dv/dt
    Compute: dot = v · a
    Compute: expected_a = μ_k · g
    Compute: ratio = |a| / expected_a
    
    IF dot < 0 AND 0.8 < ratio < 1.2:
        → SLIDING WITH FRICTION
    ELSE:
        → NO FRICTION EVENT
```

**Note:** "Friction Stop" is detected when:
- Previous state: "Sliding with Friction" (v > 0)
- Current state: "Stationary" (v ≈ 0)
- Transition occurs

### State Transitions → Motion Change

```
Detect: prev_motion, curr_motion

IF prev_motion ∈ [Constant V, Accelerating, Decelerating] 
   AND curr_motion = Stationary:
    → MOVING TO STOPPING
    
ELSE IF prev_motion = Stationary 
   AND curr_motion ∈ [Constant V, Accelerating]:
    → STATIONARY TO MOVING
```

### Interaction Events → Collisions

```
FOR each contact pair in collision:
    Compute contact normal: n̂
    Compute relative velocity before: v_rel_before = (v₁ - v₂) · n̂
    Compute relative velocity after: v_rel_after = (v₁' - v₂') · n̂
    
    IF |v_rel_before| < v_threshold:
        → NO COLLISION (negligible impact)
    ELSE:
        Compute: e = -v_rel_after / v_rel_before
        
        IF e ≥ 0.4:
            → ELASTIC COLLISION
        ELSE:
            → INELASTIC COLLISION
```

---

## References

1. **Classical Mechanics:**
   - Goldstein, H., Poole, C., Safko, J. (2002). Classical Mechanics (3rd ed.). Addison-Wesley.

2. **Collision Physics:**
   - Serway, R. A., & Jewett, J. W. (2018). Physics for Scientists and Engineers (10th ed.). Cengage.

3. **Rolling Motion:**
   - Marion, J. B., & Thornton, S. T. (2004). Classical Dynamics of Particles and Systems (5th ed.). Brooks/Cole.

4. **Friction Models:**
   - Dowson, D. (1997). History of Tribology (2nd ed.). Elsevier.


### Settings
Note: https://github.com/google-deepmind/mujoco/blob/main/doc/overview.rst#units-are-unspecified 

*MuJoCo does not specify basic physical units.*


#### Camera
- **lookat**: Point in 3D space that the camera is focused on.

    - **Format**: [x, y, z] in meters **(m)**

    - **Example**: represents the origin of the global coordinate system

- **distance**: Distance from the camera to the lookat point

    - **Unit**: meters (m)

    - **Range**: Positive values

- **azimuth**: Horizontal angle around the vertical axis (relative to the lookat point)

    - **Unit**: degrees (°)

    - **Range**: [0°, 360°] - 0° typically faces forward, increasing clockwise

- **elevation**: Vertical angle above or below the horizontal plane (relative to the lookat point)

    - **Unit**: degrees (°)

    - **Range**: [-90°, 90°] - Positive values look up, negative values look down

#### Floor
- **friction**: Describes resistive interaction between objects and the floor surface; stored as three coefficients: static, dynamic (kinetic), and rotational (rolling) friction

    - **Static**: Maximum resistive force before motion begins

        - **Unit**: Unitless (coefficient)

        - **Range**: [min_fric, max_fric]
     
        - **Physical Effect**: 	Force threshold to initiate sliding
        \[
          F_{s,\max} = \mu_s \times N = \mu_s \times (m \times g)\quad[\mathrm{N}]
        \]

    - **Dynamic (kinetic)**: Constant resistive force during sliding:
        - **Unit**: Unitless (coefficient)

        - **Range**: [min_fric, max_fric]
     
        - **Physical Effect**: 	Resistance during sliding
        \[
          F_{k} = \mu_k \times N\quad[\mathrm{N}]
        \]

     - **Rolling friction**: Resistive torque opposing rolling

        - **Unit**: Unitless (coefficient)

        - **Range**: [min_fric, max_fric]
      
        - **Physical Effect**: Resistive torque opposing rolling
       \[
          \tau_{r} = \mu_r \times N \times R\quad[\mathrm{N\cdot m}]
        \]

  - Here,  
    - \(N = m \times g\) is the normal force (kg·m/s²).  
    - \(R\) is the object’s radius (m).  








- **rgba**: Defines the color and transparency of the floor based on friction values.
        
    - **Format**: [red, green, blue, alpha]

    - **Unit**: Normalized values between 0.0 and 1.0

    - **Behavior**: Lower friction results in brighter colors; alpha is always set to 1 (fully opaque).

- **specular**: Determines how shiny or reflective the floor appears.

    - **Unit**: Normalized values between 0.0 and 1.0

    - **Behavior**: Higher friction produces less shine; lower friction results in a shinier surface.

- **shininess**: Controls the sharpness of reflections on the floor surface.

    - **Unit**: Normalized values between 0.0 and 1.0

    - **Behavior**: Higher friction produces a matte finish; lower friction results in sharper reflections.


#### Light
- **pos**: Light source position in 3D space

    - **Format**: [x, y, z] in normalized scene coordinates

    - **Range**:
        - x: [-1, 1]

        - y: [-1, 1]

        - z: [0.1, 0.7] (vertical placement above scene floor)

    - **Constraint**: Minimum distance between lights enforced via min_distance parameter

- **diffuse**: Base color of emitted light

    - **Format**: RGB values in [0.0, 1.0] range

    - **Randomization**:

        - Red: [0.7, 1.0]

        - Green/Blue: [0.8, 1.0]

    - **Effect**: Determines perceived "warmth" of light source

- **specular**: Reflection characteristics for shiny surfaces

    - **Format**: RGB values in [0.0, 1.0] range

     - **Randomization**:

        - Red: [0.5, 0.8]

        - Green/Blue: [0.5, 1.0]

    - **Behavior**: Higher values increase material reflectivity

- **cutoff**: Angular spread of light influence

    - **Unit**: Degrees (°)

     - **Range**: [0°, 180°]

    - **Purpose**: Controls light beam width and falloff

#### Objects

**Materials**:
- **metal**

- **wood**

- **rubber**

- **glass**

- **plastic**

**Shapes**:

- **ball**

- **cylinder**

- **cube**

- **block**

**Modes**:
- **collision**

- **sliding**

- **stationary**

- **offset**

**Mass** - kilograms (kg)

**Velocity** - meters per second (m/s)

- v_x: x-component of velocity - measures the rate of change of position along the x-axis

- v_y: y-component of velocity - measures the rate of change of position along the y-axis

- v_z: z-component of velocity - measures the rate of change of position along the z-axis
    
**Angular Velocity** - *radians per second **(rad/s)***, the rate of rotation of an object about an axis.

- **ω_x**: x-component of angular velocity

    - Measures the rate of rotation about the x-axis

    - Also called "roll rate" in some contexts
    
- **ω_y**: y-component of angular velocity
    
    - Measures the rate of rotation about the y-axis
    
    - Also called "pitch rate" in some contexts
    
- **ω_z**: z-component of angular velocity
    
    - Measures the rate of rotation about the z-axis
    
    - Also called "yaw rate" in some contexts

    
**Density** - Mass / Volume = kg / m³

**Elasticity** - coefficients of restitution, which are unitless values between 0 and 1. These values indicate:

- **0**: Perfectly inelastic collision (no bounce)

- **1**: Perfectly elastic collision (no energy loss)
