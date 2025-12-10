
# Physics Taxonomy Documentation

## Overview

The physics taxonomy system detects and classifies motion events, environmental interactions, and collision dynamics in simulated environments. It uses kinematic analysis backed by physics equations to categorize object behavior.

---

## 1. Kinematic Events

### 1.1 Linear Motion

#### Stationary
Object at rest with negligible velocity.

**Definition:**
$$|v| < 0.001 \text{ m/s}$$

**Detection Logic:**
Object is stationary when current velocity magnitude is near the threshold OR both current and previous velocities are below threshold.

```
vel_curr_mag = ||velocity||
vel_prev_mag = ||velocity_prev||

IF abs(vel_curr_mag - v_threshold) < v_threshold:
    → STATIONARY
ELSE IF vel_curr_mag <= v_threshold AND vel_prev_mag <= v_threshold:
    → STATIONARY
```

---

#### Constant Velocity
Object moving at fixed speed in same direction.

**Definition:**
$$|a| < 0.001 \text{ m/s}^2$$

**Detection Logic:**
Acceleration magnitude is negligible (no significant change in velocity).

```
accel = (v_curr - v_prev) / dt
a_mag = ||accel||

IF a_mag < a_threshold:
    → CONSTANT VELOCITY
```

---

#### Accelerating
Object increasing speed.

**Definition:**
$$|v_{\text{current}}| > |v_{\text{previous}}|$$

**Detection Logic:**
Current speed exceeds previous speed.

```
IF ||v_curr|| > ||v_prev||:
    → ACCELERATING
```

---

#### Decelerating
Object decreasing speed.

**Definition:**
$$|v_{\text{current}}| < |v_{\text{previous}}|$$

**Detection Logic:**
Current speed is less than previous speed.

```
IF ||v_curr|| < ||v_prev||:
    → DECELERATING
```

**Common Causes:**
- Friction: $a = -\mu_k g$
- Collision impulse
- Air resistance

---

### 1.2 Rotational Motion

#### Pure Rotation
Object rotating about its center with no linear motion.

**Definition:**
- Linear velocity near threshold: $|v| \leq 0.001$ m/s
- Significant angular velocity: $|\omega| > 0.001$ rad/s

**Detection Logic:**
```
v_mag = ||linear_velocity||
ω_mag = ||angular_velocity||

IF v_mag <= v_threshold AND ω_mag > v_threshold:
    → PURE ROTATION
```

**Examples:**
- Spinning top in place
- Ball spinning without translation

---

#### Rolling Motion
Smooth rolling without slipping (no-slip condition satisfied).

**Definition:**
$$|v - r\omega| < 0.01 \text{ m/s}$$

where $r$ is object radius, $\omega$ is angular velocity magnitude, $v$ is linear velocity magnitude.

**Detection Logic:**
```
v_mag = ||linear_velocity||
ω_mag = ||angular_velocity||
rolling_diff = v_mag - ω_mag * radius

IF abs(rolling_diff) < epsilon (0.01):
    → ROLLING MOTION
```

**Contact Point Velocity:**
$$\vec{v}_{\text{contact}} = \vec{v}_{\text{center}} - r\omega \approx 0$$

**Applicable Shapes:**
- Sphere / Ball
- Cylinder

---

#### Rolling Motion with Slipping
Rolling motion where slip velocity is non-zero.

**Definition:**
- Both linear and angular motion present: $v > 0.001$ AND $\omega > 0.001$
- Slip condition violated: $|v - r\omega| \geq 0.01$

**Detection Logic:**
```
v_mag = ||linear_velocity||
ω_mag = ||angular_velocity||

IF v_mag > v_threshold AND ω_mag > v_threshold:
    → ROLLING MOTION WITH SLIPPING
```

**Slip Velocity:**
$$v_{\text{slip}} = v - r\omega \neq 0$$

**Common Scenarios:**
- Ball skidding with rotation
- Wheel slipping on ice

---

## 2. Environmental Interactions

### 2.1 Friction-Induced Events

#### Sliding with Friction
Object moving with kinetic friction causing deceleration.

**Definition:**
1. Object is moving: $v > 0.001$ m/s
2. Deceleration opposes motion: $\vec{v} \cdot \vec{a} < 0$
3. Deceleration aligns with friction model: alignment > 0.90

**Detection Logic:**
```
v_mag = ||velocity||
a_mag = ||acceleration||
expected_friction_accel = friction_coefficient * gravity

IF v_mag > v_threshold:
    IF dot(velocity, acceleration) < 0:
        decel_vector = -velocity / v_mag * a_mag
        expected_decel_vector = -velocity / v_mag * |expected_friction_accel|
        alignment = dot(decel_vector, expected_decel_vector)
        
        IF alignment > 0.90:
            → SLIDING WITH FRICTION
```

**Kinetic Friction Equation:**
$$a = -\mu_k g$$

**Exclusion:** Not detected for shapes `ball` or `cylinder` (these primarily roll).

---

#### Friction Stop
Object has come to rest after friction deceleration.

**Definition:**
1. Object is stationary: $v < 0.001$ m/s
2. Negligible acceleration: $|a| < 0.001$ m/s²

**Detection Logic:**
```
IF v_mag <= v_threshold AND a_mag <= a_threshold:
    → FRICTION STOP
```

---

## 3. State Transitions

### 3.1 Motion Change

#### Moving to Stopping
Transition from dynamic motion to stopping (deceleration).

**Definition:**
- Previous motion state: Moving (Constant Velocity, Accelerating, or Decelerating)
- Current motion state: Decelerating

**Detection Logic:**
```
IF prev_motion_mapped == "Moving" AND curr_motion == "Decelerating":
    → MOVING TO STOPPING
```

**Motion State Mapping:**
- Stationary → "Stationary"
- Constant Velocity, Accelerating, Decelerating → "Moving"

---

#### Stationary to Moving
Transition from rest to dynamic motion.

**Definition:**
- Previous motion state: Stationary
- Current motion state: Accelerating or Constant Velocity

**Detection Logic:**
```
IF prev_motion_mapped == "Stationary" AND curr_motion in [Accelerating, Constant Velocity]:
    → STATIONARY TO MOVING
```

---

## 4. Interaction Events

### 4.1 Collisions

#### Elastic Collision
Objects collide with kinetic energy largely conserved (bouncing).

**Coefficient of Restitution:**
$$e = -\frac{(v_1' - v_2') \cdot \hat{n}}{(v_1 - v_2) \cdot \hat{n}}$$

where $\hat{n}$ is the contact normal.

**Elastic Condition:**
$$e \geq 0.4$$

**Detection Logic:**
```
rel_vel_pre = dot(v1_pre - v2_pre, normal)
rel_vel_post = dot(v1_post - v2_post, normal)

IF abs(rel_vel_pre) < v_threshold:
    → None (negligible impact)

restitution = -rel_vel_post / rel_vel_pre

IF restitution >= 0.4:
    → ELASTIC COLLISION
```

**Properties:**
- Bouncing behavior
- Minimal energy loss
- Momentum conserved (always)

---

#### Inelastic Collision
Objects collide with significant kinetic energy dissipated.

**Coefficient of Restitution:**
$$e < 0.4$$

**Detection Logic:**
```
restitution = -rel_vel_post / rel_vel_pre

IF restitution < 0.4:
    → INELASTIC COLLISION
```

**Energy Loss:**
$$\Delta E = KE_{\text{before}} - KE_{\text{after}} > 0$$

Dissipated as heat, sound, and deformation.

**Properties:**
- Absorbing impact
- Significant energy loss
- Momentum conserved (always)

---

### 4.2 Collision Analysis

Additional metrics computed during collision detection:

**Relative Velocity Magnitude:**
$$v_{\text{rel}} = ||v_1 - v_2||$$

**Relative Velocity Along Normal:**
$$v_{\text{rel,n}} = (v_1 - v_2) \cdot \hat{n}$$

**Kinetic Energy Before:**
$$KE = \frac{1}{2}m_1||v_1||^2 + \frac{1}{2}m_2||v_2||^2$$

**Head-On Collision Detection:**
Collision is head-on when both objects approach primarily along the contact normal:
$$\frac{|v_1 \cdot \hat{n}|}{||v_1||} > 0.8 \text{ AND } \frac{|v_2 \cdot \hat{n}|}{||v_2||} > 0.8$$

**Energy Transfer Classification:**
- *Elastic (Energy Conserved):* $KE_{\text{post}} / KE_{\text{pre}} > 0.9$
- *Partially Inelastic:* $0.5 < KE_{\text{post}} / KE_{\text{pre}} \leq 0.9$
- *Highly Inelastic:* $KE_{\text{post}} / KE_{\text{pre}} \leq 0.5$

**Momentum Conservation Check:**
$$\text{is\_conserved} = |1.0 - p_{\text{ratio}}| < 0.1$$

where $p_{\text{ratio}} = ||p_{\text{after}}|| / ||p_{\text{before}}||$

---

## 5. Physics Constants

| Constant | Value | Unit |
|----------|-------|------|
| Gravity | 9.81 | m/s² |
| Velocity Threshold | 0.001 | m/s |
| Acceleration Threshold | 0.001 | m/s² |
| Rolling Epsilon | 0.01 | m/s |
| Elastic Collision Factor | 0.5 | (unitless) |
| Friction Alignment Threshold | 0.90 | (unitless) |
| Precision (rounding) | 5 | decimals |

---

## 6. Complete Label Reference

| Category | Subcategory | Label | Condition |
|---|---|---|---|
| Kinematic Events | Linear Motion | Stationary | $\|v\| < 0.001$ |
| Kinematic Events | Linear Motion | Constant Velocity | $\|a\| < 0.001$ |
| Kinematic Events | Linear Motion | Accelerating | $\|v_{\text{curr}}\| > \|v_{\text{prev}}\|$ |
| Kinematic Events | Linear Motion | Decelerating | $\|v_{\text{curr}}\| < \|v_{\text{prev}}\|$ |
| Kinematic Events | Rotational Motion | Pure Rotation | $v < 0.001$ AND $\omega > 0.001$ |
| Kinematic Events | Rotational Motion | Rolling Motion | $\|v - r\omega\| < 0.01$ |
| Kinematic Events | Rotational Motion | Rolling Motion with Slipping | $v > 0.001$ AND $\omega > 0.001$ AND $\|v - r\omega\| \geq 0.01$ |
| Environmental Interactions | Friction | Sliding with Friction | $v > 0.001$ AND alignment > 0.90 |
| Environmental Interactions | Friction | Friction Stop | $v < 0.001$ AND $a < 0.001$ |
| State Transitions | Motion Change | Moving to Stopping | Transition: Moving → Decelerating |
| State Transitions | Motion Change | Stationary to Moving | Transition: Stationary → Accelerating/Constant Velocity |
| Interaction Events | Collision | Elastic Collision | $e \geq 0.4$ |
| Interaction Events | Collision | Inelastic Collision | $e < 0.4$ |


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

