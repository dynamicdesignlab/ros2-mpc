import glob
from os import path

from setuptools import setup

package_name = "rooster_viz"

setup(
    name=package_name,
    version="0.0.2",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (path.join("share", package_name, "config"), glob.glob("config/*")),
        (path.join("share", package_name, "launch"), glob.glob("launch/*")),
    ],
    install_requires=["setuptools", "transforms3d"],
    zip_safe=True,
    maintainer="Firstname Lastname",
    maintainer_email="username@domain.com",
    description="Visualization utilities for rooster package",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vehicle_visualizer=rooster_viz.vehicle_visualizer:main",
            "world_visualizer=rooster_viz.world_visualizer:main",
        ],
    },
)
