from utils.validators import *

print(is_valid_email("abc@test.com"))
print(is_valid_email("wrong-email"))

print(is_valid_trust_score(75))
print(is_valid_trust_score(120))

print(is_valid_established_year(2015))
print(is_valid_established_year(1800))

print(is_not_empty("EcoLink"))
print(is_not_empty(""))