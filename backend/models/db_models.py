from datetime import datetime
from backend.extensions import db


class PromptHistory(db.Model):
    __tablename__ = "prompt_history"

    id = db.Column(db.Integer, primary_key=True)
    operation_type = db.Column(db.String(100), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    output = db.Column(db.Text)
    output_file = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "operation_type": self.operation_type,
            "prompt": self.prompt,
            "output": self.output,
            "output_file": self.output_file,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
