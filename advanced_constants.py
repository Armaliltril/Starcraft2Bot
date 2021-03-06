from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id

_TRAIN_ZEALOT = actions.FUNCTIONS.Train_Zealot_quick.id
_TRAIN_ADEPT = actions.FUNCTIONS.Train_Adept_quick.id
_TRAIN_STALKER = actions.FUNCTIONS.Train_Stalker_quick.id
_TRAIN_HIGH_TEMPLAR = actions.FUNCTIONS.Train_HighTemplar_quick.id
_TRAIN_DARK_TEMPLAR = actions.FUNCTIONS.Train_DarkTemplar_quick.id
_TRAIN_SENTRY = actions.FUNCTIONS.Train_Sentry_quick.id
_TRAIN_PHOENIX = actions.FUNCTIONS.Train_Phoenix_quick.id
_TRAIN_CARRIER = actions.FUNCTIONS.Train_Carrier_quick.id
_TRAIN_VOID_RAY = actions.FUNCTIONS.Train_VoidRay_quick.id
_TRAIN_WARP_PRISM = actions.FUNCTIONS.Train_WarpPrism_quick.id
_TRAIN_OBSERVER = actions.FUNCTIONS.Train_Observer_quick.id
_TRAIN_IMMORTAL = actions.FUNCTIONS.Train_Immortal_quick.id
_TRAIN_PROBE = actions.FUNCTIONS.Train_Probe_quick.id

_BUILD_NEXUS = actions.FUNCTIONS.Build_Nexus_screen.id
_BUILD_PYLON = actions.FUNCTIONS.Build_Pylon_screen.id
_BUILD_ASSIMILATOR = actions.FUNCTIONS.Build_Assimilator_screen.id
_BUILD_GATEWAY = actions.FUNCTIONS.Build_Gateway_screen.id
_BUILD_FORGE = actions.FUNCTIONS.Build_Forge_screen.id
_BUILD_FLEET_BEACON = actions.FUNCTIONS.Build_FleetBeacon_screen.id
_BUILD_TWILIGHT_COUNCIL = actions.FUNCTIONS.Build_TwilightCouncil_screen.id
_BUILD_PHOTON_CANNON = actions.FUNCTIONS.Build_PhotonCannon_screen.id
_BUILD_STARGATE = actions.FUNCTIONS.Build_Stargate_screen.id
_BUILD_TEMPLAR_ARCHIVE = actions.FUNCTIONS.Build_TemplarArchive_screen.id
_BUILD_DARK_SHRINE = actions.FUNCTIONS.Build_DarkShrine_screen.id
_BUILD_ROBOTICS_BAY = actions.FUNCTIONS.Build_RoboticsBay_screen.id
_BUILD_ROBOTICS_FACILITY = actions.FUNCTIONS.Build_RoboticsFacility_screen.id
_BUILD_CYBERNETICS_CORE = actions.FUNCTIONS.Build_CyberneticsCore_screen.id


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_ = 5

# Protoss buildings
_PROTOSS_NEXUS = 59
_PROTOSS_PYLON = 60
_PROTOSS_ASSIMILATOR = 61
_PROTOSS_GATEWAY = 62
_PROTOSS_FORGE = 63
_PROTOSS_FLEET_BEACON = 64
_PROTOSS_TWILIGHT_COUNCIL = 65
_PROTOSS_PHOTON_CANNON = 66
_PROTOSS_STARGATE = 67
_PROTOSS_TEMPLATE_ARCHIVE = 68
_PROTOSS_DARK_SHRINE = 69
_PROTOSS_ROBOTIC_BAY = 70
_PROTOSS_ROBOTICS_FACILITY = 71
_PROTOSS_CYBERNETICS_CORE = 72

# Protoss units
_PROTOSS_ZEALOT = 73
_PROTOSS_STALKER = 74
_PROTOSS_HIGH_TEMPLAR = 75
_PROTOSS_DARK_TEMPLAR = 76
_PROTOSS_SENTRY = 77
_PROTOSS_PHOENIX = 78
_PROTOSS_CARRIER = 79
_PROTOSS_VOID_RAY = 80
_PROTOSS_WARP_PRISM = 81
_PROTOSS_OBSERVER = 82
_PROTOSS_IMMORTAL = 83
_PROTOSS_PROBE = 84
_PROTOSS_ADEPT = 311

_NEUTRAL_MINERAL_FIELD = 341
_NEUTRAL_VESPENE_GEYSER = 342

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'advanced_protoss_agent_data'

ACTION_DO_NOTHING = "doNothing"
ACTION_ATTACK = "attack"
ACTION_MOVE_SCREEN = "moveScreen"

ACTION_TRAIN_ZEALOT = "trainZealot"
ACTION_TRAIN_ADEPT = "trainAdept"
ACTION_TRAIN_STALKER = "trainStalker"
ACTION_TRAIN_HIGH_TEMPLAR = "trainHighTemplar"
ACTION_TRAIN_DARK_TEMPLAR = "trainDarkTemplar"
ACTION_TRAIN_SENTRY = "trainSentry"
ACTION_TRAIN_PHOENIX = "trainPhoenix"
ACTION_TRAIN_CARRIER = "trainCarrier"
ACTION_TRAIN_VOID_RAY = "trainVoidRay"
ACTION_TRAIN_WARP_PRISM = "trainWarpPrism"
ACTION_TRAIN_OBSERVER = "trainObserver"
ACTION_TRAIN_IMMORTAL = "trainImmortal"
ACTION_TRAIN_PROBE = "trainProbe"

ACTION_BUILD_NEXUS = "buildNexus"
ACTION_BUILD_PYLON = "buildPylon"
ACTION_BUILD_ASSIMILATOR = "buildAssimilator"
ACTION_BUILD_GATEWAY = "buildGateway"
ACTION_BUILD_FORGE = "buildForge"
ACTION_BUILD_FLEET_BEACON = "buildFleetBeacon"
ACTION_BUILD_TWILIGHT_COUNCIL = "buildTwilightCouncil"
ACTION_BUILD_PHOTON_CANNON = "buildPhotonCannon"
ACTION_BUILD_STARGATE = "buildStargate"
ACTION_BUILD_TEMPLAR_ARCHIVE = "buildTemplarArchive"
ACTION_BUILD_DARK_SHRINE = "buildDarkShrine"
ACTION_BUILD_ROBOTICS_BAY = "buildRoboticsBay"
ACTION_BUILD_ROBOTICS_FACILITY = "buildRoboticsFacility"
ACTION_BUILD_CYBERNETICS_CORE = "buildCyberneticsCore"

gateway_train_actions = [
    ACTION_TRAIN_ZEALOT,
    ACTION_TRAIN_ADEPT,
    ACTION_TRAIN_STALKER,
    ACTION_TRAIN_HIGH_TEMPLAR,
    ACTION_TRAIN_DARK_TEMPLAR,
    ACTION_TRAIN_SENTRY,
]

stargate_train_actions = [
    ACTION_TRAIN_PHOENIX,
    ACTION_TRAIN_CARRIER,
    ACTION_TRAIN_VOID_RAY,
]

roboticsFaCility_train_actions = [
    ACTION_TRAIN_OBSERVER,
    ACTION_TRAIN_IMMORTAL,
]

nexus_train_actions = [
    ACTION_TRAIN_PROBE,
]

train_actions = gateway_train_actions + \
                stargate_train_actions + \
                roboticsFaCility_train_actions + \
                nexus_train_actions

build_actions = [
    ACTION_BUILD_NEXUS,
    ACTION_BUILD_PYLON,
    ACTION_BUILD_ASSIMILATOR,
    ACTION_BUILD_GATEWAY,
    ACTION_BUILD_FORGE,
    ACTION_BUILD_FLEET_BEACON,
    ACTION_BUILD_TWILIGHT_COUNCIL,
    ACTION_BUILD_PHOTON_CANNON,
    ACTION_BUILD_STARGATE,
    ACTION_BUILD_TEMPLAR_ARCHIVE,
    ACTION_BUILD_DARK_SHRINE,
    ACTION_BUILD_ROBOTICS_BAY,
    ACTION_BUILD_ROBOTICS_FACILITY,
    ACTION_BUILD_CYBERNETICS_CORE,
]

smart_actions = train_actions + \
                build_actions + \
                [ACTION_DO_NOTHING, ]

action_to_function_map = {
    ACTION_TRAIN_ZEALOT: _TRAIN_ZEALOT,
    ACTION_TRAIN_ADEPT: _TRAIN_ADEPT,
    ACTION_TRAIN_STALKER: _TRAIN_STALKER,
    ACTION_TRAIN_HIGH_TEMPLAR: _TRAIN_HIGH_TEMPLAR,
    ACTION_TRAIN_DARK_TEMPLAR: _TRAIN_DARK_TEMPLAR,
    ACTION_TRAIN_SENTRY: _TRAIN_SENTRY,
    ACTION_TRAIN_PHOENIX: _TRAIN_PHOENIX,
    ACTION_TRAIN_CARRIER: _TRAIN_CARRIER,
    ACTION_TRAIN_VOID_RAY: _TRAIN_VOID_RAY,
    ACTION_TRAIN_WARP_PRISM: _TRAIN_WARP_PRISM,
    ACTION_TRAIN_OBSERVER: _TRAIN_OBSERVER,
    ACTION_TRAIN_IMMORTAL: _TRAIN_IMMORTAL,
    ACTION_TRAIN_PROBE: _TRAIN_PROBE,

    ACTION_BUILD_NEXUS: _BUILD_NEXUS,
    ACTION_BUILD_PYLON: _BUILD_PYLON,
    ACTION_BUILD_ASSIMILATOR: _BUILD_ASSIMILATOR,
    ACTION_BUILD_GATEWAY: _BUILD_GATEWAY,
    ACTION_BUILD_FORGE: _BUILD_FORGE,
    ACTION_BUILD_FLEET_BEACON: _BUILD_FLEET_BEACON,
    ACTION_BUILD_TWILIGHT_COUNCIL: _BUILD_TWILIGHT_COUNCIL,
    ACTION_BUILD_PHOTON_CANNON: _BUILD_PHOTON_CANNON,
    ACTION_BUILD_STARGATE: _BUILD_STARGATE,
    ACTION_BUILD_TEMPLAR_ARCHIVE: _BUILD_TEMPLAR_ARCHIVE,
    ACTION_BUILD_DARK_SHRINE: _BUILD_DARK_SHRINE,
    ACTION_BUILD_ROBOTICS_BAY: _BUILD_ROBOTICS_BAY,
    ACTION_BUILD_ROBOTICS_FACILITY: _BUILD_ROBOTICS_FACILITY,
    ACTION_BUILD_CYBERNETICS_CORE: _BUILD_CYBERNETICS_CORE,

    ACTION_MOVE_SCREEN: _MOVE_SCREEN
}

name_to_unit_map = {
    "nexus": _PROTOSS_NEXUS,
    "pylon": _PROTOSS_PYLON,
    "assimilator": _PROTOSS_ASSIMILATOR,
    "gateway": _PROTOSS_GATEWAY,
    "forge": _PROTOSS_FORGE,
    "fleetBeacon": _PROTOSS_FLEET_BEACON,
    "twilightCouncil": _PROTOSS_TWILIGHT_COUNCIL,
    "photonCannon": _PROTOSS_PHOTON_CANNON,
    "stargate": _PROTOSS_STARGATE,
    "templarArchive": _PROTOSS_TEMPLATE_ARCHIVE,
    "darkShrine": _PROTOSS_DARK_SHRINE,
    "roboticsBay": _PROTOSS_ROBOTIC_BAY,
    "roboticsFacility": _PROTOSS_ROBOTICS_FACILITY,
    "cyberneticsCore": _PROTOSS_CYBERNETICS_CORE,

    "zealot": _PROTOSS_ZEALOT,
    "stalker": _PROTOSS_STALKER,
    "highTemplar": _PROTOSS_HIGH_TEMPLAR,
    "darkTemplar": _PROTOSS_DARK_TEMPLAR,
    "sentry": _PROTOSS_SENTRY,
    "phoenix": _PROTOSS_PHOENIX,
    "carrier": _PROTOSS_CARRIER,
    "voidRay": _PROTOSS_VOID_RAY,
    "observer": _PROTOSS_OBSERVER,
    "immortal": _PROTOSS_IMMORTAL,
    "probe": _PROTOSS_PROBE,
    "adept": _PROTOSS_ADEPT,
}

for x in range(0, 64):
    for y in range(0, 64):
        if (x + 1) % 32 == 0 and (y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(x - 16) + '_' + str(y - 16))
            smart_actions.append(ACTION_MOVE_SCREEN + '_' + str(x - 16) + '_' + str(y - 16))
            for build_action in build_actions:
                smart_actions.append(build_action + '_' + str(x - 16) + '_' + str(y - 16))
