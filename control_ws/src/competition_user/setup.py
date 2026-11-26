from glob import glob
from setuptools import find_packages, setup

package_name = 'competition_user'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lry9640971-sudo',
    maintainer_email='292311248+lry9640971-sudo@users.noreply.github.com',
    description='Closed-loop mission planning and thrust control for autonomous USV berthing.',
    license='NOASSERTION',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ['berthing_node = competition_user.closed_loop.node:main',
        ],
    },
)
