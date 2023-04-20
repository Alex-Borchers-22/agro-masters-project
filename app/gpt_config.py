class Config(object):
    DEBUG = True
    TESTING = False

class DevelopmentConfig(Config):
    SECRET_KEY = "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"
    OPENAI_KEY = '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'

config = {
    'development': DevelopmentConfig,
    'testing': DevelopmentConfig,
    'production': DevelopmentConfig
}