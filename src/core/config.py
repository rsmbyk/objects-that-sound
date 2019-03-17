from configparser import ConfigParser, ExtendedInterpolation


class Section:
    def __init__(self, section):
        for option in section:
            setattr(self, option, section[option])


class Config:
    def __init__(self, config_filename):
        config_ = ConfigParser(interpolation=ExtendedInterpolation())
        config_.read(config_filename)
        for section in config_:
            setattr(self, section, Section(config_[section]))


config = Config('config.ini')
