from setuptools import setup, find_packages

setup(
    name='heart_plus',  # Replace with your app's name
    version='0.1',
    description='A Streamlit app for heart disease prediction',
    author='Paras Jain',  # Replace with your name
    author_email='paras2004jain@gmail.com',  # Replace with your email
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'joblib',
        'scikit-learn',
    ],
)
