class SubBridgeError(RuntimeError):
    """Base exception."""


class CodexNotInstalledError(SubBridgeError):
    pass


class CodexNotAuthenticatedError(SubBridgeError):
    pass


class CodexWrongAuthModeError(SubBridgeError):
    pass


class CodexProcessError(SubBridgeError):
    pass


class CodexTurnError(SubBridgeError):
    pass


class CodexProtocolError(SubBridgeError):
    pass


class ClaudeNotInstalledError(SubBridgeError):
    pass


class ClaudeNotAuthenticatedError(SubBridgeError):
    pass


class ClaudeWrongAuthModeError(SubBridgeError):
    pass


class ClaudeProcessError(SubBridgeError):
    pass


class ClaudeTurnError(SubBridgeError):
    pass


class ClaudeProtocolError(SubBridgeError):
    pass
