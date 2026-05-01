from pathlib import Path

from pydantic import SecretStr, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from src.utils.logger_config import logger


# this class bears all environment variables used in the application
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
        toml_file=Path(__file__).parents[2] / "config.toml",
    )

    # Optional API keys
    openai_api_key: SecretStr | None = None
    replicate_api_token: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    # Required
    meili_master_key: SecretStr = Field(
        validation_alias="master_key"
    )  # defined in config.toml

    # Has defaults, overridable
    meili_http_addr: str = Field(
        default="localhost:7700", validation_alias="http_addr"
    )  # overridden via config.toml
    port: int = 8080  # overridden via .env or system environment

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        _ = init_settings
        _ = file_secret_settings
        # reference: https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/#changing-priority
        return (
            env_settings,  # highest priority
            dotenv_settings,  # .env file
            TomlConfigSettingsSource(settings_cls),  # config.toml as fallback
        )

    def get_openai_api_key(self) -> str:
        """Returns the OpenAI API key if it exists, otherwise exits process."""
        if self.openai_api_key is None:
            logger.critical(
                "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
            )
            exit(1)
        return self.openai_api_key.get_secret_value()

    def get_replicate_api_token(self) -> str:
        """Returns the Replicate API token if it exists, otherwise exits process."""
        if self.replicate_api_token is None:
            logger.critical(
                "Replicate API token is not set. Please set the REPLICATE_API_TOKEN environment variable."
            )
            exit(1)
        return self.replicate_api_token.get_secret_value()

    def get_groq_api_key(self) -> str:
        """Returns the Groq API key if it exists, otherwise exits process."""
        if self.groq_api_key is None:
            logger.critical(
                "Groq API key is not set. Please set the GROQ_API_KEY environment variable."
            )
            exit(1)
        return self.groq_api_key.get_secret_value()

    def get_hf_token(self) -> str | None:
        """Returns the Hugging Face token if it exists, otherwise returns None."""
        return self.hf_token.get_secret_value() if self.hf_token is not None else None


settings = Settings()  # type: ignore

if __name__ == "__main__":
    print(settings.model_dump())
