from typing import Tuple, Type

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class MusicModelSettings(BaseSettings):
    path_backend: str = "./backend_data"
    backend_host: str = "127.0.0.1"
    backend_port: int = 22448


class EdaSettings(BaseSettings):
    min_num_genres: int = 40


class Settings(BaseSettings, case_sensitive=False):
    music_model: MusicModelSettings = MusicModelSettings()
    model_config = SettingsConfigDict(yaml_file="config/config.yml")
    eda_config: EdaSettings = EdaSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


config = Settings()  # type: ignore
