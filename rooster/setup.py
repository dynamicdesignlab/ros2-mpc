import glob
from os import path

from setuptools import setup

package_name = "rooster"

setup(
    name=package_name,
    version="0.0.2",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            path.join("share", package_name, "config/vehicles"),
            glob.glob("config/vehicles/*"),
        ),
        (
            path.join("share", package_name, "config/worlds"),
            glob.glob("config/worlds/*"),
        ),
        (path.join("share", package_name, "launch"), glob.glob("launch/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Firstname Lastname",
    maintainer_email="username@domain.com",
    description="NMPC for vehicle control",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "coordinator=rooster.coordinator:main",
            "mpc_runner=rooster.mpc_runner:main",
            "simulator=rooster.simulator:main"
        ],
    },
)
