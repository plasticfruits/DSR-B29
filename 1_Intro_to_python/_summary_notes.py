# Sumary Notes
### Into to python w/ Rachel Berryman
### 10.01.2022

# Basics
#- suggestion: create new environment for each new project
#- add ! to run a bash command in python
!python version
dir(x) # returns all properties and methods of the specified object

# check function
range?

## data types
type(16) # Int
type(16.0) # float

### STRINGS

# string formating
print("Today is day: {}".format(day))
print(f"Today is day: {day}")

#format decimal places
x = 1.00015
print(f"{x:.4f}, {x:.2f}, {x:.1f}, {x:.0f}")

# string stripping
"this is a string".replace("this", "that")
"this is a string".replace(" ", "")

new_string = "hello8"
"8" in new_string

# Conditional statements
if x=10:
    print("10")
elif x <10:
    print("smaller")
else:
    print("grater")


# Comparisons
5 != 6
5 == 5.0 # they are equivalent

# Logical Oeprators
True and True
True & True
True | True # or
True is not True


# List []
# %%
my_list = [1, 2, "hi", True, 6.666] # can be mixed
new_list = list(np.array([1,2])) # change type

my_list[0:4] # 4 excluded
[my_list[1], my_list[4]] # two elements
my_list[-1] # takes last element of list

my_list[2][0] # returns the h from hi


# Loops
first = [2,4,8]

for i in first:
    print(i * 3)


























#
