import ast
import flexidep
from dafne_dl.model_loaders import generic_load_model
import dill

def parse_complex_options(options_string):
    def parse_value(value):
        value = value.strip()
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    options = {}
    current_key = None
    current_value = ''
    stack = []
    in_quotes = False
    quote_char = None

    for char in options_string:
        if char in ('"', "'") and (not in_quotes or char == quote_char):
            in_quotes = not in_quotes
            quote_char = char if in_quotes else None
            current_value += char
        elif char == ',' and not stack and not in_quotes:
            if current_key:
                options[current_key] = parse_value(current_value)
                current_key = None
                current_value = ''
        elif char == '=' and not stack and not in_quotes:
            current_key = current_value.strip()
            current_value = ''
        elif char == '[' and not in_quotes:
            stack.append('[')
            current_value += char
        elif char == ']' and not in_quotes:
            if stack and stack[-1] == '[':
                stack.pop()
            current_value += char
        else:
            current_value += char

    if current_key:
        options[current_key] = parse_value(current_value)

    return options


def dictionary_to_options(d):
    """
    Convert a dictionary to a string format that can be parsed by parse_complex_options.

    Args:
        d: Dictionary to convert

    Returns:
        String representation that parse_complex_options can convert back to the original dictionary
    """

    def format_value(value):
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float, list, tuple, dict)):
            return repr(value)
        elif isinstance(value, str):
            # Check if the string needs to be quoted
            if (',' in value or '=' in value or
                    '[' in value or ']' in value or
                    value.strip().lower() in ('true', 'false') or
                    (value.strip() and (value.strip()[0].isdigit() or value.strip()[0] in '+-'))):
                # Use repr to get proper quoting, but fix double quotes
                return repr(value)
            return value
        else:
            return str(value)

    parts = []
    for key, value in d.items():
        formatted_value = format_value(value)
        parts.append(f"{key}={formatted_value}")

    return ','.join(parts)

def ensure_dependencies_from_metadata(metadata, APP_STRING='network.dafne.dafne_models'):

    # check and install model dependencies
    dependencies = metadata.get('dependencies', {})
    if not dependencies: return

    dependency_manager = flexidep.DependencyManager(
        config_file=None,
        config_string=None,
        unique_id=APP_STRING,
        interactive_initialization=False,
        use_gui=False,
        install_local=False,
        package_manager=flexidep.PackageManagers.pip,
        extra_command_line='',
    )

    for package, alternative_str in dependencies.items():
        print("Processing package", package)
        dependency_manager.process_single_package(package, alternative_str)

def get_medicalvolume_orientation_from_metadata(metadata):
    # check if the model has a specific orientation
    model_orientation = metadata.get('orientation', None)

    if isinstance(model_orientation, str):
        # the orientation is a string (Axial/Transversal, Sagittal, Coronal)
        model_orientation = model_orientation.lower()
        if model_orientation.startswith('a') or model_orientation.startswith('t'):
            model_orientation = ('LR', 'AP', 'SI')
        elif model_orientation.startswith('s'):
            model_orientation = ('AP', 'IS', 'LR')
        elif model_orientation.startswith('c'):
            model_orientation = ('LR', 'SI', 'AP')
        else:
            print("Unknown orientation")
            model_orientation = None

    return model_orientation

def ensure_compatible_orientation_inplace(image, metadata):
    original_orientation = image.orientation
    model_orientation = get_medicalvolume_orientation_from_metadata(metadata)

    if model_orientation is not None and model_orientation != original_orientation:
        image.reformat(model_orientation, inplace=True)

def load_model_and_install_deps(model_path):
    with open(model_path, 'rb') as f:
        input_dict = dill.load(f)

    metadata = input_dict.get('metadata', {})

    ensure_dependencies_from_metadata(metadata)

    # build the model object. Now that the dependencies are installed, the model can be loaded
    model = generic_load_model(input_dict)

    return model
