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
