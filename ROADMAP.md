# Roadmap

The local gateway (`subbridge.use_subscription()`, `subbridge run`, `subbridge serve`) already lets code written for the official `anthropic` and `openai` SDKs run on your subscription for prototyping. See the [README](README.md#use-the-official-sdks).

## Planned

- Tool calling (function calling) through the gateway, so SDK code that defines tools runs unchanged while prototyping. Today the gateway rejects `tools`, `tool_choice`, and related parameters with a 400.
- Image and document input through the gateway, for requests that include them. Today the gateway rejects image and document content with a 400.
- More OpenAI and Anthropic request features the gateway currently rejects with a 400, such as `previous_response_id`, where the CLIs can support them.

Items have no dates. Order reflects current priority and can change.

## Suggest something

Open an [issue](https://github.com/kunjan-p/subbridge/issues) describing what you want to build and where SubBridge gets in the way. Pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
