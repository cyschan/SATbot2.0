import copy


def validate_parameters(required_arguments, default_arguments, parameter_dictionary):
    new_parameter_dictionary = copy.deepcopy(parameter_dictionary)

    # For each required argument in required_arguments, we need to raise an error if not in the dictionary
    for argument in required_arguments:
        if argument not in new_parameter_dictionary:
            raise ValueError('Required parameter ({}) missing from parameter dictionary.'.format(argument))

    # For each tuple (argument, default value) set this in the parameter dictionary
    for argument, value in default_arguments:
        if argument not in new_parameter_dictionary:
            new_parameter_dictionary[argument] = value

    return new_parameter_dictionary