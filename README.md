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

#### Camera
- **lookat**: Point in 3D space that the camera is focused on.

    - **Format**: [x, y, z] in meters **(m)**

    - **Example**: represents the origin of the global coordinate system

- **distance**: Distance from the camera to the lookat point

    - **Unit**: meters (m)

    - **Range**: Positive values

- **azimuth**: Horizontal angle around the vertical axis

    - **Unit**: degrees (°)

    - **Range**: [0°, 360°] - 0° typically faces forward, increasing clockwise

- **elevation**: Vertical angle above or below the horizontal plane

    - **Unit**: degrees (°)

    - **Range**: [-90°, 90°] - Positive values look up, negative values look down