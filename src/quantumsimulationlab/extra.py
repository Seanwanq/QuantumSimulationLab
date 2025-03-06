def nameof(var):
    for name, value in globals().items():
        if value is var:
            return name
