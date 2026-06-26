"""
Autonomous Spot firefighter controller.

State machine:
  STAND_UP  – initial stand-up transition (blocking, once at startup)
  SEEK      – no fire known; hold still until receiver delivers a target
  APPROACH  – diagonal trot toward the fire with real-time heading correction
  THROW     – water delivered; stand still through cooldown then return to SEEK

Communication:
  Receiver "receiver"  – receives {"fires": [[x,y,z], ...]} from the fire
                         supervisor every ~80 ms.
  customData           – Spot writes the water quantity; the supervisor reads
                         it and spawns a physics water ball aimed in Spot's
                         forward (-Z local) direction.
"""

import json
import math

from controller import Robot


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


class Spot(Robot):
    NUMBER_OF_JOINTS = 12

    # ---- trot gait ----
    GAIT_FREQ  = 1.5    # Hz — stride frequency
    STRIDE_AMP = 0.35   # rad — shoulder rotation amplitude at full forward scale
    LIFT_AMP   = 0.15   # rad — elbow lift during swing phase

    # ---- firefighting ----
    STOP_RANGE    = 4.5   # m  — stop & throw water when horizontally within this range
    WATER_QTY     = 10    # supervisor creates one water ball of this effective size
    COOLDOWN_SECS = 12.0  # s  — wait after throw before seeking the next fire

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())

        motor_names = [
            "front left shoulder abduction motor",
            "front left shoulder rotation motor",
            "front left elbow motor",
            "front right shoulder abduction motor",
            "front right shoulder rotation motor",
            "front right elbow motor",
            "rear left shoulder abduction motor",
            "rear left shoulder rotation motor",
            "rear left elbow motor",
            "rear right shoulder abduction motor",
            "rear right shoulder rotation motor",
            "rear right elbow motor",
        ]
        self.motors = [self.getDevice(n) for n in motor_names]

        # Sensors (GPS and InertialUnit are built into the Spot proto)
        self.gps      = self.getDevice("gps")
        self.imu      = self.getDevice("inertial unit")
        self.receiver = self.getDevice("receiver")
        self.keyboard = self.getKeyboard()

        self.gps.enable(self.time_step)
        self.imu.enable(self.time_step)
        self.receiver.enable(self.time_step)
        self.keyboard.enable(10 * self.time_step)

        # Autonomous state
        self.fire_target   = None   # [x, y, z] of nearest known fire
        self.state         = "STAND_UP"
        self._cooldown_end = 0.0
        self._water_steps  = 0     # remaining steps to keep the water command active

        print("Spot: autonomous firefighter ready — walking toward fires.")

    # ═══════════════════════════════════════════════════ gait helpers ═══

    def _stand_still(self):
        stand = [-0.1, 0.0, 0.0,  0.1, 0.0, 0.0,  -0.1, 0.0, 0.0,  0.1, 0.0, 0.0]
        for i, v in enumerate(stand):
            self.motors[i].setPosition(v)

    def _trot(self, left_amp, right_amp):
        """
        Apply one time-step of a diagonal trot gait.

        left_amp / right_amp: signed stride scale for each side [-1, +1].
        Positive = walk forward.  Negative = walk backward (for in-place turns).
        Diagonal pairs: FL+RR share phase_a, FR+RL share phase_b = -phase_a.
        """
        phase = 2.0 * math.pi * self.GAIT_FREQ * self.getTime()
        pa = math.sin(phase)             # FL + RR
        pb = math.sin(phase + math.pi)  # FR + RL
        A, L = self.STRIDE_AMP, self.LIFT_AMP

        # FL (0,1,2) — left side, pair A
        self.motors[0].setPosition(-0.1)
        self.motors[1].setPosition(A * pa * left_amp)
        self.motors[2].setPosition(-L if pa * left_amp > 0 else 0.05)

        # FR (3,4,5) — right side, pair B
        self.motors[3].setPosition(0.1)
        self.motors[4].setPosition(A * pb * right_amp)
        self.motors[5].setPosition(-L if pb * right_amp > 0 else 0.05)

        # RL (6,7,8) — left side, pair B
        self.motors[6].setPosition(-0.1)
        self.motors[7].setPosition(A * pb * left_amp)
        self.motors[8].setPosition(-L if pb * left_amp > 0 else 0.05)

        # RR (9,10,11) — right side, pair A
        self.motors[9].setPosition(0.1)
        self.motors[10].setPosition(A * pa * right_amp)
        self.motors[11].setPosition(-L if pa * right_amp > 0 else 0.05)

    # ═══════════════════════════════════════════════ receiver / fire data ═══

    def _drain_receiver(self):
        """
        Process all queued packets and keep the closest fire as the target.
        An empty fires list means all fires are extinguished; clears the target.
        """
        pos = self.gps.getValues()
        while self.receiver.getQueueLength() > 0:
            try:
                data  = json.loads(self.receiver.getData().decode())
                fires = data.get("fires", [])
                if fires:
                    self.fire_target = min(
                        fires,
                        key=lambda f: (f[0] - pos[0])**2 + (f[1] - pos[1])**2
                    )
                else:
                    self.fire_target = None
            except Exception:
                pass
            self.receiver.nextPacket()

    # ═══════════════════════════════════════════════════ navigation ═══

    def _dist_xy(self):
        """Horizontal (XY) distance to the current fire target."""
        if self.fire_target is None:
            return float('inf')
        p = self.gps.getValues()
        dx = self.fire_target[0] - p[0]
        dy = self.fire_target[1] - p[1]
        return math.sqrt(dx * dx + dy * dy)

    def _heading_error(self):
        """
        Signed angle error [rad] between Spot's current yaw and the bearing
        to the fire.  Positive = fire is to the left -> turn left.
        """
        p = self.gps.getValues()
        bearing = math.atan2(self.fire_target[1] - p[1],
                             self.fire_target[0] - p[0])
        yaw = self.imu.getRollPitchYaw()[2]
        return (bearing - yaw + math.pi) % (2.0 * math.pi) - math.pi

    # ═══════════════════════════════════════════ stand-up helper (blocking) ═══

    def movementDecomposition(self, target, duration):
        n     = int(duration * 1000 / self.time_step)
        cur   = [m.getTargetPosition() for m in self.motors]
        diffs = [(target[i] - cur[i]) / n for i in range(self.NUMBER_OF_JOINTS)]
        for _ in range(n):
            for j in range(self.NUMBER_OF_JOINTS):
                cur[j] += diffs[j]
                self.motors[j].setPosition(cur[j])
            self.step(self.time_step)

    # ══════════════════════════════════════════════════════════ main loop ═══

    def run(self):
        # One-time blocking stand-up before entering the event loop
        self.movementDecomposition(
            [-0.1, 0.0, 0.0,  0.1, 0.0, 0.0,  -0.1, 0.0, 0.0,  0.1, 0.0, 0.0],
            duration=1.5
        )
        self.state = "SEEK"

        while self.step(self.time_step) != -1:

            # ── manual keyboard override ─────────────────────────────────
            if self.keyboard.getKey() == ord('D'):
                self._water_steps = 1

            # ── water command to supervisor ──────────────────────────────
            if self._water_steps > 0:
                self.setCustomData(str(self.WATER_QTY))
                self._water_steps -= 1
            else:
                self.setCustomData("0")

            # ── receive latest fire broadcast ────────────────────────────
            self._drain_receiver()
            dist = self._dist_xy()
            now  = self.getTime()

            # ── state machine ────────────────────────────────────────────
            if self.state == "SEEK":
                self._stand_still()
                if self.fire_target is not None:
                    print("Spot: fire at ({:.1f}, {:.1f}), {:.1f} m away — approaching".format(
                        self.fire_target[0], self.fire_target[1], dist))
                    self.state = "APPROACH"

            elif self.state == "APPROACH":
                if self.fire_target is None:
                    print("Spot: fire extinguished — resuming search")
                    self.state = "SEEK"
                elif dist <= self.STOP_RANGE:
                    self._stand_still()
                    self._water_steps  = 3          # hold for 3 steps so supervisor sees it
                    self._cooldown_end = now + self.COOLDOWN_SECS
                    self.state = "THROW"
                    print("Spot: throwing water at ({:.1f}, {:.1f}), dist={:.1f} m".format(
                        self.fire_target[0], self.fire_target[1], dist))
                else:
                    # Heading-corrected diagonal trot
                    err   = self._heading_error()
                    turn  = clamp(err, -1.0, 1.0)
                    # Reduce forward speed when facing well away from target
                    fwd   = max(0.35, 1.0 - abs(turn) * 0.55)
                    left  = clamp(fwd - turn * 0.65, -1.0, 1.0)
                    right = clamp(fwd + turn * 0.65, -1.0, 1.0)
                    self._trot(left, right)

            elif self.state == "THROW":
                self._stand_still()
                if now >= self._cooldown_end:
                    print("Spot: cooldown complete — resuming search")
                    self.fire_target = None
                    self.state = "SEEK"


robot = Spot()
robot.run()
