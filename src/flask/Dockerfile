FROM python:3.7-slim

# Set the working directory
WORKDIR /

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000


ENTRYPOINT [ "python", "app.py" ]