package com.volleycut.nativeanalysis

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Matches the web header without placing the white splash artwork on Paper. */
@Composable
internal fun VolleySpliceBrand(compact: Boolean = false) {
    Row(
        modifier = Modifier.clearAndSetSemantics { contentDescription = "VolleySplice" },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Image(
            painter = painterResource(R.drawable.volleysplice_icon),
            contentDescription = null,
            modifier = Modifier.size(if (compact) 28.dp else 36.dp),
        )
        if (!compact) {
            Text(
                text = buildAnnotatedString {
                    append("volley")
                    withStyle(SpanStyle(color = Color(0xFF0875F5))) { append("splice") }
                },
                color = Color(0xFF052B50),
                fontSize = 20.sp,
                fontStyle = FontStyle.Italic,
                fontWeight = FontWeight.Black,
                letterSpacing = (-0.75).sp,
                maxLines = 1,
                softWrap = false,
            )
        }
    }
}
