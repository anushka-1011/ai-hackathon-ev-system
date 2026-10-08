# Site-level coordinates used for GPS-based station optimization.
#
# Caltech and JPL coordinates are based on published ACN-related
# charging-cluster location data.
#
# Office001 is anonymized in ACN-Data as an office building
# in the Silicon Valley / Northern California area, so its
# coordinate is an approximate reference point.

SITE_LOCATIONS = {
    "caltech": {
        "name": "Caltech",
        "latitude": 34.134765,
        "longitude": -118.127183,
        "coordinate_type": "approximate_site_cluster",
    },

    "jpl": {
        "name": "JPL",
        "latitude": 34.198841,
        "longitude": -118.170402,
        "coordinate_type": "approximate_site_cluster",
    },

    "office001": {
        "name": "Office001",
        "latitude": 37.3382,
        "longitude": -121.8863,
        "coordinate_type": "approximate_silicon_valley_reference",
    },
}