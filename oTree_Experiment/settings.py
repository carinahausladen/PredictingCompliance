from os import environ

# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

SESSION_CONFIG_DEFAULTS = {
    'real_world_currency_per_point': 1,
    'participation_fee': 0,
    'doc': "",
}

SESSION_CONFIGS = [
    {
        'name': 'FHM',
        'display_name': "FHM",
        'num_demo_participants': 2,
        'app_sequence': ['hausladen_FHM']
    },
]


LANGUAGE_CODE = 'de'
REAL_WORLD_CURRENCY_CODE = 'EUR'
USE_POINTS = False

ROOMS = [
    dict(
        name='Xlab',
        display_name='Xlab@FUB',
        participant_label_file='Xlab.txt'
    ),
]

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')
DEMO_PAGE_INTRO_HTML = """ """

SECRET_KEY = '=s#7s$6+9hrhhsh+x$#u_awkkic$m_h@6g0duap!j%))xsmjr*'

# if an app is included in SESSION_CONFIGS, you don't need to list it here
INSTALLED_APPS = ['otree']
