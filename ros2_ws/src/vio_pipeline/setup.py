from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'vio_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
        ('share/' + package_name + '/launch',
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='VIO Dev',
    maintainer_email='user@example.com',
    description='VIO pipeline for EuRoC MAV dataset',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_estimation_node = vio_pipeline.vio_node:main',
            'tf_publisher = vio_pipeline.tf_publisher:main',
            'feature_tracking_node = vio_pipeline.feature_tracking_node:main',
            'ground_truth_publisher = vio_pipeline.ground_truth_publisher:main',
        ],
    },
)
