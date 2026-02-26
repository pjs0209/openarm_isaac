from setuptools import find_packages, setup

package_name = 'openarm_converters'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Enactic, Inc.',
    maintainer_email='openarm_dev@enactic.ai',
    description='JointState to Action converters for OpenArm',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_to_real_converter = openarm_converters.sim_to_real_converter:main',
            'real_to_sim_converter = openarm_converters.real_to_sim_converter:main',
        ],
    },
)
