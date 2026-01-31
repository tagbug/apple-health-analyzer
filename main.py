from src.cli import main
from src.i18n import Translator, resolve_locale

if __name__ == "__main__":
  try:
    main()
  except Exception as error:
    translator = Translator(resolve_locale())
    message = str(error)
    if message.startswith("config.error."):
      parts = message.split(":")
      key = parts[0]
      payload = parts[1:]
      if key == "config.error.export_xml_not_found" and payload:
        message = translator.t(key, path=payload[0])
      elif key == "config.error.export_xml_not_file" and payload:
        message = translator.t(key, path=payload[0])
      elif key == "config.error.export_xml_not_readable" and payload:
        message = translator.t(key, path=payload[0])
      elif key == "config.error.output_not_writable" and len(payload) >= 2:
        message = translator.t(key, path=payload[0], error=": ".join(payload[1:]))
      elif key == "config.error.invalid_log_level" and len(payload) >= 2:
        message = translator.t(key, value=payload[0], options=payload[1])
      elif key == "config.error.invalid_locale" and len(payload) >= 2:
        message = translator.t(key, value=payload[0], options=payload[1])
      else:
        try:
          message = translator.t(key)
        except Exception:
          message = translator.t("cli.common.unexpected_error")
    print(message)
