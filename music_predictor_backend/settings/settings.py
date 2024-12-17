from typing import Type, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, YamlConfigSettingsSource


class MusicModelSettings(BaseSettings):
    path_backend: str = "./backend_data"

class Settings(BaseSettings, case_sensitive=False):
    music_model: MusicModelSettings = MusicModelSettings()
    model_config = SettingsConfigDict(yaml_file="config/config.yml")

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
