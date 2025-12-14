# Physics Formulas and Model Testing

This document provides a comprehensive reference for all physics formulas used in the PHLOP system, explaining why each formula is used, what physical concepts they demonstrate, and how they test model understanding through question generation.

---

## Table of Contents

1. [Linear Motion](#linear-motion)
2. [Rotational Motion](#rotational-motion)
3. [Collision Physics](#collision-physics)
4. [Friction Physics](#friction-physics)
5. [State Transitions](#state-transitions)

---

## Linear Motion

### Velocity Magnitude

**Formula:**
$$|v| = \sqrt{v_x^2 + v_y^2 + v_z^2}$$

**Why We Use This Formula:**

Velocity magnitude provides a scalar measure of an object's speed, independent of direction. This is essential for:
- **Motion Classification**: Determining if an object is stationary, moving slowly, or moving fast
- **Speed Comparisons**: Comparing velocities before and after events (collisions, friction stops)
- **Threshold Detection**: Identifying when objects transition between motion states

**Physical Concept:**

The magnitude of velocity represents the instantaneous speed of an object. In 3D space, velocity is a vector quantity, but its magnitude gives us the scalar speed that determines kinetic energy and momentum magnitudes.

**What This Tests in Models:**

- **Speed Understanding**: Can the model understand that speed is the magnitude of velocity?
- **Motion Detection**: Can the model identify when objects are moving vs. stationary?
- **Causal Reasoning**: Can the model connect velocity changes to physical causes (collisions, friction)?

**Used in Questions:**
- Causal questions compare speeds before and after collisions to test understanding of momentum transfer
- Motion detection questions test whether models can identify when objects start or stop moving

---

### Acceleration

**Formula:**
$$\vec{a} = \frac{\vec{v}_{\text{current}} - \vec{v}_{\text{previous}}}{\Delta t}$$

**Magnitude:**
$$|a| = ||\vec{a}||$$

**Why We Use This Formula:**

Acceleration quantifies how velocity changes over time, which is fundamental to:
- **Motion State Classification**: Distinguishing between constant velocity, accelerating, and decelerating motion
- **Force Detection**: Identifying when forces are acting (acceleration ≠ 0 implies net force)
- **Friction Analysis**: Detecting deceleration patterns that match friction models

**Physical Concept:**

Acceleration is the rate of change of velocity. When acceleration is zero, velocity is constant (Newton's first law). When acceleration is non-zero, a net force is acting (Newton's second law: $F = ma$).

**Derivation:**

From the definition of velocity:
$$v(t) = \frac{dx}{dt}$$

Acceleration is the derivative of velocity:
$$a(t) = \frac{dv}{dt} = \frac{d^2x}{dt^2}$$

For discrete time steps:
$$a \approx \frac{\Delta v}{\Delta t} = \frac{v(t + \Delta t) - v(t)}{\Delta t}$$

**What This Tests in Models:**

- **Acceleration Understanding**: Can the model recognize that acceleration means velocity is changing?
- **Force Reasoning**: Can the model connect acceleration to forces (friction, collisions)?
- **Motion Classification**: Can the model distinguish between different types of motion (constant, accelerating, decelerating)?

**Used in Questions:**
- Counterfactual questions use acceleration to reason about friction effects
- Motion classification questions test understanding of acceleration vs. constant velocity

---

## Rotational Motion

### Rolling Condition (No-Slip Condition)

**Formula:**
$$v = r \omega$$

where:
- $v$ = linear velocity magnitude
- $r$ = object radius
- $\omega$ = angular velocity magnitude

**Detection Condition:**
$$|v - r\omega| < \epsilon$$

where $\epsilon = 0.01$ m/s is the tolerance threshold.

**Why We Use This Formula:**

The rolling condition distinguishes between pure rolling (no slipping) and rolling with slipping:
- **Pure Rolling**: The contact point between object and surface has zero velocity relative to the surface
- **Rolling with Slipping**: There is relative motion at the contact point, indicating kinetic friction

**Physical Concept:**

For an object rolling without slipping, the linear velocity of the center of mass equals the tangential velocity at the rim. This occurs when:
$$\vec{v}_{\text{contact}} = \vec{v}_{\text{center}} - r\omega \hat{\theta} = 0$$

The condition $v = r\omega$ ensures that the point of contact is instantaneously at rest relative to the surface, eliminating kinetic friction at the contact point.

**Derivation:**

For a rolling object, the velocity of any point is the sum of translational and rotational components:
$$\vec{v}_{\text{point}} = \vec{v}_{\text{CM}} + \vec{\omega} \times \vec{r}_{\text{point}}$$

At the contact point (lowest point on the object):
$$\vec{v}_{\text{contact}} = v_{\text{CM}} \hat{x} - r\omega \hat{x} = (v_{\text{CM}} - r\omega) \hat{x}$$

For no-slip rolling, $\vec{v}_{\text{contact}} = 0$, therefore:
$$v_{\text{CM}} = r\omega$$

**What This Tests in Models:**

- **Rotational Motion Understanding**: Can the model understand the relationship between linear and angular velocity?
- **Rolling vs. Slipping**: Can the model distinguish between pure rolling and rolling with slipping?
- **Physical Intuition**: Can the model reason about why rolling objects move differently than sliding objects?

**Used in Questions:**
- Rolling detection questions count objects exhibiting rolling motion
- Future questions could test understanding of the rolling condition itself

---

## Collision Physics

### Kinetic Energy

**Formula (Single Object):**
$$KE = \frac{1}{2}mv^2$$

**Formula (Two Objects):**
$$KE_{\text{total}} = \frac{1}{2}m_1v_1^2 + \frac{1}{2}m_2v_2^2$$

**Energy Ratio:**
$$KE_{\text{ratio}} = \frac{KE_{\text{after}}}{KE_{\text{before}}}$$

**Energy Loss Percentage:**
$$\text{Energy Loss \%} = (1 - KE_{\text{ratio}}) \times 100\%$$

**Why We Use This Formula:**

Kinetic energy quantifies the energy of motion and is crucial for:
- **Collision Classification**: Distinguishing elastic (energy conserved) from inelastic (energy lost) collisions
- **Energy Transfer Analysis**: Understanding how energy is distributed or dissipated during collisions
- **Physical Reasoning**: Testing whether models understand that energy can be lost (converted to heat, sound, deformation)

**Physical Concept:**

Kinetic energy represents the work required to accelerate an object to its current speed. In elastic collisions, kinetic energy is conserved. In inelastic collisions, some kinetic energy is converted to other forms (thermal energy, sound, deformation).

**Derivation:**

From work-energy theorem:
$$W = \Delta KE = \int F \cdot dx$$

For constant force:
$$W = F \cdot x = ma \cdot x$$

Using kinematic equation $v^2 = v_0^2 + 2ax$:
$$W = m \cdot \frac{v^2 - v_0^2}{2x} \cdot x = \frac{1}{2}mv^2 - \frac{1}{2}mv_0^2$$

Therefore:
$$KE = \frac{1}{2}mv^2$$

**What This Tests in Models:**

- **Energy Conservation Understanding**: Can the model calculate kinetic energy from mass and velocity?
- **Energy Loss Recognition**: Does the model understand that energy can be lost in inelastic collisions?
- **Quantitative Reasoning**: Can the model compute energy percentages and ratios?
- **Physical Intuition**: Does the model understand that bouncing (elastic) vs. sticking (inelastic) relates to energy conservation?

**Used in Questions:**
- Energy conservation questions test understanding of kinetic energy in collisions
- Questions ask models to reason about energy loss percentages
- Models must distinguish between elastic and inelastic collisions based on energy behavior

---

### Momentum

**Formula (Single Object):**
$$\vec{p} = m\vec{v}$$

**Formula (Two Objects):**
$$\vec{p}_{\text{total}} = m_1\vec{v}_1 + m_2\vec{v}_2$$

**Momentum Conservation Ratio:**
$$p_{\text{ratio}} = \frac{|\vec{p}_{\text{after}}|}{|\vec{p}_{\text{before}}|}$$

**Conservation Check:**
$$\text{is\_conserved} = |1.0 - p_{\text{ratio}}| < 0.1$$

**Why We Use This Formula:**

Momentum is always conserved in collisions (in the absence of external forces), making it a fundamental quantity for:
- **Collision Analysis**: Verifying that momentum conservation holds (it always should in isolated systems)
- **Velocity Prediction**: Using momentum conservation to predict post-collision velocities
- **Physical Reasoning**: Testing whether models understand that momentum is a conserved quantity

**Physical Concept:**

Momentum is conserved in all collisions due to Newton's third law (action-reaction pairs). Unlike energy, momentum is always conserved, even in inelastic collisions. This makes momentum conservation a fundamental principle of collision physics.

**Derivation:**

From Newton's second law:
$$F = \frac{dp}{dt}$$

For a collision between two objects:
$$F_{1 \to 2} = -F_{2 \to 1}$$

Therefore:
$$\frac{dp_1}{dt} = -\frac{dp_2}{dt}$$

$$\frac{d(p_1 + p_2)}{dt} = 0$$

This means $p_1 + p_2$ is constant, so momentum is conserved.

**What This Tests in Models:**

- **Momentum Conservation Understanding**: Does the model know that momentum is always conserved in collisions?
- **Momentum Calculation**: Can the model calculate momentum from mass and velocity?
- **Conservation Recognition**: Can the model identify when momentum conservation holds (it should always hold in isolated collisions)?
- **Physical Principles**: Does the model understand the fundamental difference between energy (not always conserved) and momentum (always conserved)?

**Used in Questions:**
- Momentum conservation questions test whether models understand that momentum is conserved
- Questions verify that models can calculate momentum ratios
- Models must recognize that momentum conservation is a fundamental principle

---

### Coefficient of Restitution

**Formula:**
$$e = -\frac{(\vec{v}_1' - \vec{v}_2') \cdot \hat{n}}{(\vec{v}_1 - \vec{v}_2) \cdot \hat{n}}$$

where:
- $\hat{n}$ = contact normal vector
- $\vec{v}_1, \vec{v}_2$ = pre-collision velocities
- $\vec{v}_1', \vec{v}_2'$ = post-collision velocities

**Relative Velocity Along Normal:**
$$v_{\text{rel,n}} = (\vec{v}_1 - \vec{v}_2) \cdot \hat{n}$$

**Why We Use This Formula:**

The coefficient of restitution quantifies the "bounciness" of a collision:
- **Collision Classification**: $e = 1$ for perfectly elastic, $e = 0$ for perfectly inelastic
- **Energy Relationship**: Related to energy conservation (higher $e$ means less energy loss)
- **Collision Prediction**: Can predict post-collision velocities when combined with momentum conservation

**Physical Concept:**

The coefficient of restitution measures how much of the relative velocity along the contact normal is restored after collision. It depends on material properties and collision geometry.

**Classification:**
- **Elastic Collision**: $e \geq 0.5$ (high bounce, energy largely conserved)
- **Inelastic Collision**: $e < 0.5$ (low bounce, significant energy loss)

**What This Tests in Models:**

- **Collision Understanding**: Can the model understand the relationship between bounce behavior and energy?
- **Material Properties**: Does the model recognize that different materials have different coefficients of restitution?
- **Collision Geometry**: Can the model understand how collision angle affects the coefficient of restitution?

**Used in Questions:**
- Indirectly used through collision classification
- Questions about elastic vs. inelastic collisions test understanding of bounce behavior

---

### Relative Velocity

**Formula:**
$$\vec{v}_{\text{rel}} = \vec{v}_1 - \vec{v}_2$$

**Magnitude:**
$$|\vec{v}_{\text{rel}}| = ||\vec{v}_1 - \vec{v}_2||$$

**Relative Velocity Along Normal:**
$$v_{\text{rel,n}} = (\vec{v}_1 - \vec{v}_2) \cdot \hat{n}$$

**Why We Use This Formula:**

Relative velocity is essential for:
- **Collision Severity**: Higher relative velocity means more severe collision
- **Head-on Detection**: Determines if collision is head-on (both velocities aligned with normal) or glancing
- **Energy Analysis**: Relative velocity magnitude relates to collision energy

**Physical Concept:**

Relative velocity represents how fast two objects are approaching each other. In head-on collisions, both objects' velocities are primarily along the contact normal, leading to maximum energy transfer.

**Head-on Collision Detection:**
$$\frac{|\vec{v}_1 \cdot \hat{n}|}{|\vec{v}_1|} > 0.8 \text{ AND } \frac{|\vec{v}_2 \cdot \hat{n}|}{|\vec{v}_2|} > 0.8$$

**What This Tests in Models:**

- **Collision Geometry Understanding**: Can the model understand how collision angle affects outcomes?
- **Relative Motion**: Does the model understand that relative velocity determines collision severity?
- **Head-on vs. Glancing**: Can the model distinguish between different collision types?

**Used in Questions:**
- Stored in collision context for future use
- Could be used to test understanding of collision geometry and relative motion

---

## Friction Physics

### Friction Acceleration

**Formula:**
$$a_{\text{friction}} = -\mu_k g$$

where:
- $\mu_k$ = kinetic friction coefficient
- $g$ = gravitational acceleration (9.81 m/s²)
- Negative sign indicates deceleration (opposes motion)

**Why We Use This Formula:**

Friction acceleration is fundamental to:
- **Friction Detection**: Identifying when objects are decelerating due to friction
- **Motion Prediction**: Predicting how objects will slow down on surfaces
- **Counterfactual Reasoning**: Understanding how changing friction affects motion

**Physical Concept:**

Kinetic friction opposes motion with a force proportional to the normal force. For objects on horizontal surfaces, the normal force equals weight ($mg$), so friction force is $F_f = \mu_k mg$. By Newton's second law:
$$a = \frac{F}{m} = \frac{-\mu_k mg}{m} = -\mu_k g$$

**Derivation:**

From Newton's second law:
$$F_{\text{net}} = ma$$

For sliding object with friction:
$$F_{\text{net}} = -F_{\text{friction}} = -\mu_k N$$

For horizontal surface, normal force equals weight:
$$N = mg$$

Therefore:
$$ma = -\mu_k mg$$
$$a = -\mu_k g$$

**What This Tests in Models:**

- **Friction Understanding**: Can the model understand that friction causes constant deceleration?
- **Force-Motion Connection**: Does the model connect friction force to deceleration?
- **Quantitative Reasoning**: Can the model calculate friction acceleration from coefficient and gravity?

**Used in Questions:**
- Counterfactual questions test understanding of how friction affects motion
- Questions ask what happens when friction coefficient changes

---

### Stopping Time (Friction)

**Formula:**
$$t_{\text{stop}} = \frac{v_0}{\mu_k g}$$

where:
- $v_0$ = initial velocity
- $\mu_k$ = friction coefficient
- $g$ = gravitational acceleration

**Why We Use This Formula:**

Stopping time quantifies how long it takes for friction to bring an object to rest:
- **Counterfactual Reasoning**: Understanding how friction changes affect stopping time
- **Motion Duration**: Predicting how long objects will slide before stopping
- **Inverse Relationship**: Demonstrating that stopping time is inversely proportional to friction

**Physical Concept:**

Under constant deceleration $a = -\mu_k g$, velocity decreases linearly:
$$v(t) = v_0 + at = v_0 - \mu_k g t$$

Setting $v(t) = 0$:
$$0 = v_0 - \mu_k g t_{\text{stop}}$$
$$t_{\text{stop}} = \frac{v_0}{\mu_k g}$$

**Derivation:**

From kinematic equation with constant acceleration:
$$v(t) = v_0 + at$$

With friction acceleration $a = -\mu_k g$:
$$v(t) = v_0 - \mu_k g t$$

At stopping time, $v(t_{\text{stop}}) = 0$:
$$0 = v_0 - \mu_k g t_{\text{stop}}$$
$$t_{\text{stop}} = \frac{v_0}{\mu_k g}$$

**Counterfactual Reasoning:**

If friction coefficient doubles:
$$t_{\text{new}} = \frac{v_0}{2\mu_k g} = \frac{1}{2} \cdot \frac{v_0}{\mu_k g} = \frac{t_{\text{original}}}{2}$$

Therefore, doubling friction halves stopping time.

**What This Tests in Models:**

- **Counterfactual Reasoning**: Can the model reason about what happens when physical parameters change?
- **Inverse Relationships**: Does the model understand that stopping time is inversely proportional to friction?
- **Quantitative Prediction**: Can the model predict how friction changes affect motion duration?
- **Physical Intuition**: Does the model understand that more friction means faster stopping?

**Used in Questions:**
- Counterfactual friction questions test understanding of friction-motion relationships
- Questions ask: "If friction doubles, what happens to stopping time?"
- Models must reason about inverse relationships between friction and duration

---

## State Transitions

### Motion State Detection

**States:**
- **Stationary**: $|v| \leq 0.001$ m/s
- **Accelerating**: $|a| \geq 0.001$ m/s² AND velocity increasing
- **Decelerating**: $|a| \geq 0.001$ m/s² AND velocity decreasing
- **Constant Velocity**: $|a| < 0.001$ m/s²

**State Transitions:**
- **Moving → Stopping**: Previous state was "Moving" AND current state is "Decelerating"
- **Stationary → Moving**: Previous state was "Stationary" AND current state is "Accelerating" or "Constant Velocity"

**Why We Use This Formula:**

State transitions identify when objects change their fundamental motion state:
- **Causal Reasoning**: Identifying what causes objects to start or stop moving
- **Event Detection**: Recognizing significant physical events (collisions, friction stops)
- **Temporal Reasoning**: Understanding cause-effect relationships over time

**Physical Concept:**

Motion states represent fundamental categories of object behavior. Transitions between states indicate physical events:
- **Stationary → Moving**: Force applied (collision, push)
- **Moving → Stopping**: Friction, collision, or other decelerating force

**What This Tests in Models:**

- **Causal Understanding**: Can the model identify what causes motion changes?
- **Temporal Reasoning**: Does the model understand cause-effect relationships over time?
- **State Recognition**: Can the model recognize different motion states?
- **Event Detection**: Can the model identify significant physical events (collisions, stops)?

**Used in Questions:**
- Causal questions test understanding of what causes objects to start moving
- Questions ask: "Was the collision the cause of motion?" requiring temporal and causal reasoning
- Models must connect events (collisions) to state changes (stationary → moving)

---

## Summary: What Each Concept Tests in Models

### Energy Conservation Understanding
- **Tests**: Can the model calculate kinetic energy? Does it understand energy loss in inelastic collisions? Can it compute energy percentages?
- **Questions**: Energy conservation questions in collisions
- **Key Insight**: Energy is not always conserved (unlike momentum)

### Momentum Conservation Understanding
- **Tests**: Does the model know momentum is always conserved? Can it calculate momentum from mass and velocity? Does it understand momentum conservation ratio?
- **Questions**: Momentum conservation questions in collisions
- **Key Insight**: Momentum is always conserved in isolated collisions

### Collision Physics Understanding
- **Tests**: Understanding of relative velocity, head-on vs. glancing collisions, energy transfer mechanisms
- **Questions**: Collision classification and analysis questions
- **Key Insight**: Collision type (elastic/inelastic) depends on energy conservation, not momentum

### Rotational Motion Understanding
- **Tests**: Understanding of rolling condition $v = r\omega$, distinction between pure rolling and rolling with slipping, relationship between linear and angular velocity
- **Questions**: Rolling detection questions
- **Key Insight**: Rolling requires specific relationship between linear and angular motion

### Friction Physics Understanding
- **Tests**: Understanding of friction acceleration $a = -\mu g$, relationship between friction coefficient and stopping time, counterfactual reasoning about friction changes
- **Questions**: Counterfactual friction questions
- **Key Insight**: Friction causes constant deceleration, stopping time inversely proportional to friction

### Causal Reasoning
- **Tests**: Can the model identify causes of motion changes? Does it understand temporal cause-effect relationships?
- **Questions**: Direct causation questions, state transition questions
- **Key Insight**: Events (collisions) cause state changes (stationary → moving)

### Counterfactual Reasoning
- **Tests**: Can the model reason about what happens when physical parameters change? Does it understand inverse relationships?
- **Questions**: Counterfactual friction questions
- **Key Insight**: Changing friction coefficient changes stopping time inversely

---

## Physics Constants

| Constant | Value | Unit | Purpose |
|----------|-------|------|---------|
| Gravity ($g$) | 9.81 | m/s² | Gravitational acceleration for friction calculations |
| Velocity Threshold | 0.001 | m/s | Minimum velocity to be considered moving |
| Acceleration Threshold | 0.001 | m/s² | Minimum acceleration for motion classification |
| Rolling Epsilon ($\epsilon$) | 0.01 | m/s | Tolerance for rolling condition detection |
| Elastic Collision Factor | 0.5 | (unitless) | Threshold for elastic collision classification |
| Momentum Conservation Tolerance | 0.1 | (unitless) | Tolerance for momentum conservation check (10%) |
| Energy Conservation Threshold | 0.9 | (unitless) | Threshold for elastic collision (90% energy conserved) |

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
