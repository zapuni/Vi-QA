"""
src/pipelines/teacher_data/validator.py
========================================
Validate output theo Output Profile Contract.
Hỗ trợ self-correction prompt khi validation fail.
"""

from __future__ import annotations

from src.pipelines.teacher_data.models import OUTPUT_PROFILES, TeacherOutput


class OutputValidator:
    """Validate output theo profile contract."""

    def validate_or_raise(self, output: TeacherOutput) -> dict[str, bool]:
        """
        Kiểm tra output theo profile.

        Raises:
            ValueError nếu thiếu required field hoặc có forbidden field.

        Returns:
            Dict validation results.
        """
        profile = output.output_profile
        contract = OUTPUT_PROFILES.get(profile)

        if not contract:
            raise ValueError(f"Unknown profile: {profile}")

        validation: dict[str, bool] = {}

        # Kiểm tra required fields
        for fld in contract["required"]:
            value = getattr(output, fld, None)
            if fld == "grounding_boxes":
                is_valid = isinstance(value, list) and len(value) > 0
            elif fld == "evidence_items":
                is_valid = isinstance(value, list) and len(value) > 0
            else:
                is_valid = value is not None and str(value).strip() != ""

            if not is_valid:
                raise ValueError(
                    f"Profile '{profile}' bắt buộc phải có field '{fld}', "
                    f"nhưng nhận được: {value!r}"
                )
            validation[f"{fld}_present"] = True

        # Kiểm tra forbidden fields
        for fld in contract["forbidden"]:
            value = getattr(output, fld, None)
            if fld == "grounding_boxes":
                has_value = isinstance(value, list) and len(value) > 0
            else:
                has_value = value is not None and str(value).strip() != ""

            if has_value:
                raise ValueError(
                    f"Profile '{profile}' KHÔNG được có field '{fld}', "
                    f"nhưng đã nhận: {value!r}"
                )
            validation[f"{fld}_absent"] = True

        # Kiểm tra bbox format (4 hoặc 5 phần tử)
        if output.grounding_boxes:
            for box in output.grounding_boxes:
                if len(box) == 4:
                    if not all(0 <= coord <= 1000 for coord in box):
                        raise ValueError(f"Bbox coords phải normalized [0-1000]: {box}")
                elif len(box) == 5:
                    # [img_idx, x1, y1, x2, y2]
                    if box[0] < 0:
                        raise ValueError(f"img_idx phải >= 0: {box}")
                    if not all(0 <= c <= 1000 for c in box[1:]):
                        raise ValueError(f"Bbox coords phải normalized [0-1000]: {box}")
                else:
                    raise ValueError(
                        f"Bbox phải có 4 hoặc 5 phần tử, nhận được {len(box)}: {box}"
                    )
            validation["bbox_format_ok"] = True

        # Kiểm tra final_answer không trống
        if not output.final_answer or output.final_answer.strip() == "":
            raise ValueError("final_answer không được trống")
        validation["final_answer_ok"] = True

        # Kiểm tra PoT code nếu có
        if output.python_code:
            if "print(" not in output.python_code and "return" not in output.python_code:
                raise ValueError("Python code phải có print() hoặc return")
            validation["python_code_ok"] = True

        return validation

    def validate_with_retry_prompt(
        self, output: TeacherOutput
    ) -> tuple[bool, str | None]:
        """
        Validate và trả về self-correction prompt nếu sai.

        Returns:
            (True, None) nếu valid.
            (False, correction_prompt) nếu invalid.
        """
        try:
            self.validate_or_raise(output)
            return True, None
        except ValueError as e:
            profile = output.output_profile
            contract = OUTPUT_PROFILES.get(profile, {})
            correction_prompt = (
                f"Lỗi format: {e}\n\n"
                f"Vui lòng sửa lại output theo đúng profile '{profile}':\n"
                f"- Required fields: {contract.get('required', [])}\n"
                f"- Forbidden fields: {contract.get('forbidden', [])}\n\n"
                f"Hãy trả lại đầy đủ theo format XML tags đã yêu cầu."
            )
            return False, correction_prompt
