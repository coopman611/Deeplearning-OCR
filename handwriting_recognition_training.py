try:
	with tf.device('/gpu:0'):
		history = model.fit(
			ds,
			validation_data=validation_ds,
			epochs=epochs,
			callbacks=[edit_distance_callback, cp_callback],
		)
except:
	history = model.fit(
			ds,
			validation_data=validation_ds,
			epochs=epochs,
			callbacks=[edit_distance_callback, cp_callback],
		)