from enum import Enum
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlatformState(Enum):
    HEALTHY      = "healthy"
    DEGRADED     = "degraded"
    DRIFTING     = "drifting"
    RETRAINING   = "retraining"
    ROLLING_BACK = "rolling_back"
    RECOVERING   = "recovering"
    CRITICAL     = "critical"


# Valid state transitions
TRANSITIONS = {
    PlatformState.HEALTHY:      [PlatformState.DEGRADED, PlatformState.DRIFTING, PlatformState.CRITICAL],
    PlatformState.DEGRADED:     [PlatformState.HEALTHY, PlatformState.CRITICAL, PlatformState.ROLLING_BACK, PlatformState.RETRAINING],
    PlatformState.DRIFTING:     [PlatformState.HEALTHY, PlatformState.RETRAINING, PlatformState.CRITICAL],
    PlatformState.RETRAINING:   [PlatformState.HEALTHY, PlatformState.DEGRADED, PlatformState.CRITICAL],
    PlatformState.ROLLING_BACK: [PlatformState.HEALTHY, PlatformState.CRITICAL],
    PlatformState.RECOVERING:   [PlatformState.HEALTHY, PlatformState.CRITICAL],
    PlatformState.CRITICAL:     [PlatformState.RECOVERING, PlatformState.ROLLING_BACK],
}


class StateMachine:

    def __init__(self):
        self.state = PlatformState.HEALTHY
        self.previous_state = None
        self.history = []
        self.state_entered_at = datetime.utcnow()

    def transition(self, new_state: PlatformState, reason: str = "") -> bool:
        allowed = TRANSITIONS.get(self.state, [])

        if new_state not in allowed:
            logger.warning(
                f"Invalid transition: {self.state.value} → {new_state.value} | Reason: {reason}"
            )
            return False

        self.previous_state = self.state
        self.state = new_state
        self.state_entered_at = datetime.utcnow()

        entry = {
            "from": self.previous_state.value,
            "to": new_state.value,
            "reason": reason,
            "timestamp": self.state_entered_at.isoformat()
        }
        self.history.append(entry)

        logger.info(
            f"STATE | {self.previous_state.value} → {new_state.value} | {reason}"
        )
        return True

    def current(self) -> str:
        return self.state.value

    def time_in_state(self) -> float:
        return (datetime.utcnow() - self.state_entered_at).total_seconds()

    def get_history(self) -> list:
        return self.history

    def is_stable(self) -> bool:
        return self.state == PlatformState.HEALTHY

    def is_busy(self) -> bool:
        return self.state in [
            PlatformState.RETRAINING,
            PlatformState.ROLLING_BACK,
            PlatformState.RECOVERING
        ]