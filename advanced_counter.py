class Counter:
    def __init__(self):
        self.nexus = 0
        self.assimilator = 0
        self.gateway = 0
        self.forge = 0
        self.fleetBeacon = 0
        self.twilightCouncil = 0
        self.photonCannon = 0
        self.stargate = 0
        self.templarArchive = 0
        self.darkShrine = 0
        self.roboticsBay = 0
        self.roboticsFacility = 0
        self.cyberneticsCore = 0

        self.zealot = 0
        self.adept = 0
        self.stalker = 0
        self.highTemplar = 0
        self.darkTemplar = 0
        self.sentry = 0
        self.phoenix = 0
        self.voidRay = 0
        self.carrier = 0
        self.observer = 0
        self.immortal = 0

        self.probe = 0

    def getValues(self):
        observations = dict()
        for attr, value in self.__dict__.items():
            observations[attr] = value
        values = []
        for key in sorted(observations):
            values.append(observations[key])
        return values

    def set(self, name, value):

        if name == "zealot":
            self.zealot = value
        elif name == "adept":
            self.adept = value
        elif name == "stalker":
            self.stalker = value
        elif name == "highTemplar":
            self.highTemplar = value
        elif name == "darkTemplar":
            self.darkTemplar = value
        elif name == "sentry":
            self.sentry = value
        elif name == "phoenix":
            self.phoenix = value
        elif name == "voidRay":
            self.voidRay = value
        elif name == "carrier":
            self.carrier = value
        elif name == "observer":
            self.observer = value
        elif name == "immortal":
            self.immortal = value
        elif name == "probe":
            self.probe = value
        elif name == "nexus":
            self.nexus = value
        elif name == "pylon":
            self.pylon = value
        elif name == "assimilator":
            self.assimilator = value
        elif name == "gateway":
            self.gateway = value
        elif name == "forge":
            self.forge = value
        elif name == "fleetBeacon":
            self.fleetBeacon = value
        elif name == "twilightCouncil":
            self.twilightCouncil = value
        elif name == "photonCannon":
            self.photonCannon = value
        elif name == "stargate":
            self.stargate = value
        elif name == "templarArchive":
            self.templarArchive = value
        elif name == "darkShrine":
            self.darkShrine = value
        elif name == "roboticsFacility":
            self.roboticsFacility = value
        elif name == "roboticsBay":
            self.roboticsBay = value
        elif name == "cyberneticsCore":
            self.cyberneticsCore = value
        else:
            raise RuntimeWarning("Counter set unexpected value: " + name)
