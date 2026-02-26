import os
from glob import glob
from setuptools import find_packages, setup

package_name = "vio_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*launch*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="dev@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "backend = vio_pkg.backend:main",
            "frontend = vio_pkg.frontend:main",
            "frontend_raft = vio_pkg.frontend_raft:main",
            "ground_truth_pub = vio_pkg.ground_truth_pub:main",
            "trajectory_comparator = vio_pkg.trajectory_comparator:main",
            "tf_broadcaster = vio_pkg.tf_broadcaster:main",
        ],
    },
)
