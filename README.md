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
- **friction**: Describes resistive interaction between objects and the floor surface; stored as three coefficients: static, dynamic (kinetic), and rotational (rolling) friction

    - **Static**: Maximum resistive force before motion begins

        - **Unit**: Unitless (coefficient)
     
        - **Physical Effect**: 	Force threshold to initiate sliding
        $$F_{s,\max} = \mu_s \times N = \mu_s \times (m \times g)\quad[\mathrm{N}]$$

    - **Dynamic (kinetic)**: Constant resistive force during sliding:
        - **Unit**: Unitless (coefficient)
     
        - **Physical Effect**: 	Resistance during sliding
        $$F_{k} = \mu_k \times N\quad[\mathrm{N}]$$

     - **Rolling friction**: Resistive torque opposing rolling

        - **Unit**: Unitless (coefficient)
      
        - **Physical Effect**: Resistive torque opposing rolling
       $$\tau_{r} = \mu_r \times N \times R\quad[\mathrm{N\cdot m}]$$

  - Here,  
    - **Normal force** (\(N\)): The reactive force perpendicular to the contact surface, supporting the object’s weight.  
        - **Definition**:  
            \[
            N = m \times g
            \]
            where  
            - \(m\) is the object’s mass (kg)  
            - \(g\) is gravitational acceleration (≈ 9.81 m/s²)  

        - **Units**: Newtons (N)  

     
    - \(R\) is the object’s radius (m).  


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

- **Physical Meaning**: Resistance to acceleration (inertia)

- **Units**: kg (derived from density × volume)

- **Newton's 2nd Law**: Directly in F=ma





**Velocity** - meters per second (m/s)
- **Physical Meaning**: Rate of change of position/orientation

- **Newton's 1st Law Connection**: Objects in motion stay in motion unless acted upon

- v_x: x-component of velocity - measures the rate of change of position along the x-axis

- v_y: y-component of velocity - measures the rate of change of position along the y-axis

- v_z: z-component of velocity - measures the rate of change of position along the z-axis
    
**Angular Velocity** - *radians per second **(rad/s)***, the rate of rotation of an object about an axis.

- **Physical Meaning**: Rate of change of position/orientation

- **Newton's 1st Law Connection**: Objects in motion stay in motion unless acted upon

- **ω_x**: x-component of angular velocity

    - Measures the rate of rotation about the x-axis

    - Also called "roll rate" in some contexts
    
- **ω_y**: y-component of angular velocity
    
    - Measures the rate of rotation about the y-axis
    
    - Also called "pitch rate" in some contexts
    
- **ω_z**: z-component of angular velocity
    
    - Measures the rate of rotation about the z-axis
    
    - Also called "yaw rate" in some contexts

    
**Density** - Mass / Volume = kg / m³:

- **Physical Meaning**: Mass per unit volume

- **Units**: kg/m³ (scaled in simulation)

- **Newton's 2nd Law Connection**: Affects mass (m = ρV) thus inertia (F=ma)


**Elasticity(Restitution Coefficient)**: coefficients of restitution, which are unitless values between 0 and 1. 

- **Physical Meaning**: Ratio of velocities after/before collision

- **Newton's 3rd Law Connection**: Governs how collision forces interact

- **Energy Interpretation**: Determines kinetic energy conservation in collisions

- These values indicate:

    - **0**: Perfectly inelastic collision (no bounce)

    - **1**: Perfectly elastic collision (no energy loss)
