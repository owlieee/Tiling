# Given an array of positive integers, write a function which returns all the unique pairs which add (equal) up to 100.
#
# Example data:
#
# sample_data = [0, 1, 100, 99, 0, 10, 90, 30, 55, 33, 55, 75, 50, 51, 49, 50, 51, 49, 51]
# sample_output = [[1,99], [0,100], [10,90], [51,49], [50,50]]

def add_to_100(sample_data):
    output_list = []
    for ix, value in enumerate(sample_data):
        for other_value in sample_data[ix+1:]:
            if value + other_value ==100:
                possible_pair = [value, other_value]
                possible_pair.sort()
                if possible_pair not in output_list:
                    output_list.append(possible_pair)
    return output_list

if __name__=='__main__':
    result = add_to_100([0, 1, 100, 99, 0, 10, 90, 30, 55, 33, 55, 75, 50, 51, 49, 51, 49, 51])
    print result
