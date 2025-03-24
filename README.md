# PHLOP Project
**P**hysics-grounded **H**ierarchical **L**atent space for **O**bject state and **P**resentation

![PHLOP](assets/kikis-delivery-service-tired.gif)

| Level 1 (General Category)    | Level 2 (Event Type)        | Level 3 (Specific Event Labels)               |
|------------------------------- |---------------------------- |----------------------------------------------- |
| **Kinematic Events**          | Linear Motion               | Constant Velocity                             |
|                               |                             | Decelerating                                  |
|                               |                             | Accelerating                                  |
|                               |                             | Stationary                                    |
|                               | ~~Projectile Motion~~           | ~~With Air Resistance~~                           |
|                               |                             | ~~Without Air Resistance~~                       |
|                               | Rotational Motion           | Pure Rotation                                 |
|                               |                             | Rolling Motion                                |
|                               |                             | Rolling Motion With Slipping                  |
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
- **friction**: Describes the interaction between objects and the floor surface

    - **friction_static**: Coefficient of static friction (resistance to initial motion).

        - **Unit**: Unitless (coefficient)

        - **Range**: [min_fric, max_fric]

    - **friction_dynamic**: Coefficient of dynamic friction (resistance during motion).

        - **Unit**: Unitless (coefficient)

        - **Range**: [min_fric, max_fric]

     - **friction_spin**: Coefficient of  rotational friction (resistance to spinning motion).

        - **Unit**: Unitless (coefficient)

        - **Range**: [min_fric, max_fric]

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
