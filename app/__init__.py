import os
from flask import Flask
from flask_cors import CORS


def create_app(config=None):
    """
    Create and configure the Flask application

    Args:
        config (dict, optional): Configuration dictionary for the app

    Returns:
        Configured Flask application instance
    """
    # Initialize the Flask app
    app = Flask(__name__)

    # Enable CORS for broader compatibility
    CORS(app)

    # Default configurations
    app.config.update({
        'UPLOAD_FOLDER': 'uploads',
        'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50 MB max file size
        'SECRET_KEY': os.urandom(24),  # Generate a secure random secret key
        'DEBUG': False
    })

    # Override with any provided configuration
    if config:
        app.config.update(config)

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Import and register blueprints or routes if needed
    try:
        # Example of registering a blueprint (uncomment and modify as needed)
        # from .routes import main_blueprint
        # app.register_blueprint(main_blueprint)
        pass
    except ImportError as e:
        app.logger.warning(f"Could not import routes: {e}")

    return app