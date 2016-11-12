
class BoundingBox:
	class InvalidBox(Exception):
		pass

	def __init__(box,x1=-1,y1=-1,x2=-1,y2=-1):
		box.min_x = float(x1)
		box.min_y = float(y1)
		box.max_x = float(x2)
		box.max_y = float(y2)

	@staticmethod
	def from_tuple_wh(t):
		if len(t) <> 4:
			print 'from_tuple_wh: requires 4-tuple!'
		else:
			min_x = float(t[0])
			min_y = float(t[1])
			max_x = min_x + float(t[2])
			max_y = min_y + float(t[3])
			return BoundingBox(min_x, min_y, max_x, max_y)

	def __str__(box):
		return 'Bounding Box - (%d,%d,%d,%d)' %\
			 (box.min_x, box.min_y, box.max_x, box.max_y)

	def w(box):
		return int(round(box.max_x - box.min_x))

	def h(box):
		return int(round(box.max_y - box.min_y))

	def ar(box):
		return float(box.h()) / box.w()

	def scale(box, factor):
		if factor < 0:
			raise InvalidBox("factor needs to be positive.")
		box.min_x = factor * box.min_x
		box.min_y = factor * box.min_y
		box.max_x = factor * box.max_x
		box.max_y = factor * box.max_y

	def pad(box, pad_x=0, pad_y=0):
		box.min_x = box.min_x - pad_x
		box.max_x = box.max_x + pad_x
		box.min_y = box.min_y - pad_y
		box.max_y = box.max_y + pad_y
		box.validate()

	def as_tuple_wh(box):
		return map(lambda x: int(round(x)),
				(box.min_x,
				box.min_y,
				box.max_x-box.min_x,
				box.max_y-box.min_y))

	def as_tuple_xyxy(box):
		return map(lambda x: int(round(x)),
				(box.min_x,
				box.min_y,
				box.max_x,
				box.max_y))

	def validate(box):
		if box.min_x >= box.max_x:
			raise InvalidBox("Invalid Box - invalid x")
		elif box.min_y >= box.max_y:
			raise InvalidBox("Invalid Box - invalid y")
		else:
			return True

	def adapt_aspect_ratio(box, target_aspect_ratio, adapt_x=True):
		"""adapt the width according to target aspect ratio
		- only - adapt the width
		"""
		if not box.validate():
			raise InvalidBox("Invalid Box")
		if not type(target_aspect_ratio) is float:
			raise InvalidBox("target_aspect_ratio needs to be float")
		w = box.w()
		h = box.h()
		if adapt_x:
			delta = 0.5 * (h / target_aspect_ratio - w )
			box.min_x = box.min_x - delta
			box.max_x = box.max_x + delta
		else:
			raise InvalidBox('only adapt width is implemented...')
