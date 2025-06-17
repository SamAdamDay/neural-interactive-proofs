Self-hosting Language Models (``nip.language_model_server``)
============================================================

.. currentmodule:: nip.language_model_server


Overview
--------

The language model server provides a way to host open-weight language models. It is
designed so that interacting with it is similar to using the OpenAI API. The main caveat
is that inference and training need to be done with separate services.

The language model server consists of the following components:

- A vLLM server that provides inference access to a model, with an OpenAI-compatible
  API.
- A training service which is accessed using a subset of the OpenAI API.
- A manager service which starts and stops the vLLM server, allowing for easy switching
  between models.

The manager service and inference service use the same port, and have the endpoints
documented below. The vLLM server runs on its own port, so must be accessed using a
different client.


Server API
----------

.. http:post:: /vllm/start

    Start the vLLM server with the specified model.
    
    :reqjson string model_name: The name of the model to serve with vLLM. Must be a valid model
        name that vLLM can load.
    :resjson string message: A message indicating that the vLLM server has started.
    :resjson string model_name: The name of the model that was started.
    :resjson int port: The port that the vLLM server is running on.


.. http:post:: /vllm/stop

    Stop the vLLM server.
    
    :reqjson bool ignore_not_running: If True, will not raise an error if the server is
        not running.
    :reqjson float timeout: The timeout in seconds to wait for the server to terminate gracefully.


.. http:get:: /vllm/status

    Get the status of the vLLM server.
    
    :resjson string status: The status of the vLLM server, which can be one of:
        
        - "online": The server is running and accepting connections.
        - "not_started": The server has not been started.
        - "crashed": The server has exited unexpectedly.
        - "not_accepting_connections": The server is running but not accepting
          connections. This can happen if the server is still starting up or if it has
          crashed.
        - "server_error": A 5xx error occurred when trying to connect to the server.
        - "other_error": Any other error occurred when trying to connect to the server.

    :resjson string error: An error message if the server is not online.


Modules
-------

.. autosummary::
   :toctree: generated/modules
   :recursive:

   server
   client
   types
   exceptions
