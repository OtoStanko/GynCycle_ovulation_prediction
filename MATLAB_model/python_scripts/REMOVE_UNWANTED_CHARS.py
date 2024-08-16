def process_file(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            # Find the position of the first "%" and ";"
            percent_pos = line.find('%')
            semicolon_pos = line.find(';')

            # Determine the position to cut the line
            cut_pos = min(pos for pos in [percent_pos, semicolon_pos])
            if percent_pos == 0:
                continue
            elif cut_pos == -1:
                outfile.write(line)
            else:
                # Write the line up to the first occurrence of "%" or ";" 
                outfile.write(line[:cut_pos+1].rstrip() + '\n')

input_filename = 'Simulation.txt'
output_filename = 'Simulation_reduced.txt'
process_file(input_filename, output_filename)
