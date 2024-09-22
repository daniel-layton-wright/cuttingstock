import unittest
import tempfile
import subprocess


class TestCuttingStock(unittest.TestCase):
    def test_simple_example(self):
        # Define the input data
        input_data = (
            "10\n"
            "3 2\n"
            "2 2\n"
        )

        # Define the expected output
        expected_output = (
            "1.0\n"
            "1\n"
            "\n"
            "2\n"
            "2\n"
        )

        with (tempfile.NamedTemporaryFile(mode='w+', delete=True) as input_file,
              tempfile.NamedTemporaryFile(mode='r+', delete=True) as output_file):
                 
            # Write input data to the temporary input file
            input_file.write(input_data)
            input_file.flush()

            # Run the cutting_stock.py script
            subprocess.run(['python', 'cutting_stock.py', input_file.name, output_file.name], check=True)

            # Read the output data
            output_file.seek(0)
            output_data = output_file.read()

            # Compare the output data to the expected output
            self.assertEqual(output_data.strip(), expected_output.strip())

if __name__ == '__main__':
    unittest.main()
