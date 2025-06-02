"""A server and client for hosting language models.

This module provides a server which does the following two things:

- It allows starting and stopping a VLLM server, and choosing the model served.
- It implements DPO fine-tuning for language models using the TRL library. This part
  exposes and API with a subset of the endpoints used by the OpenAI API, so it can be
  used as a drop-in replacement for the OpenAI API for fine-tuning language models with
  DPO.

There is also a client which can be used to interact with VLLM side of the server.
"""
