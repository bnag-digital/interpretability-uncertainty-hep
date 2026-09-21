class IncompatibleFeatureSpace(ValueError):
    """Two importance vectors describe different features, so tau is undefined."""


class InsufficientCapability(RuntimeError):
    """The supplied artifacts cannot measure the level of variation asked for."""


class LowSurrogateFidelity(RuntimeError):
    """A surrogate did not reproduce the model it is standing in for."""


class RankInputError(ValueError):
    """A rank vector was passed where importance values were expected."""
