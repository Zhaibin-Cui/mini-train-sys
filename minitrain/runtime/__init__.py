"""Runtime glue for experiments.

The runtime layer is intentionally thin: it reads config dictionaries, chooses
devices, and builds model/operator/distributed objects. It should not contain
kernel code or training algorithm details.
"""
